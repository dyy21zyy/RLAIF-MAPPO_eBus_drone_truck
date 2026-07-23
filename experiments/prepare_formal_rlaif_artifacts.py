"""Prepare and validate formal four-agent RLAIF artifacts.

This entry point is fail-closed: formal mode never fabricates labels or
checkpoints. It requires real preference JSONL generated from the configured
external evaluator and then reuses the canonical reward-model trainer/loader.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
from typing import Any
import yaml
from evaluation.scenario_bank import sha256_file
from experiments.train_multi_agent_reward_models import main as train_reward_main
from rlaif.runtime_agent_reward_model import RuntimeAgentRewardModel
from training.event_schema import REQUIRED_EVENT_COVERAGE, AGENT_TYPES

PLACEHOLDERS = ("REPLACE_WITH", "PLACEHOLDER", "MISSING_FORMAL", "TBD", "UNKNOWN")


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _hash_json(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def verify_evaluator_config(cfg: dict[str, Any]) -> dict[str, str]:
    ev = cfg.get("evaluator", {})
    api_key_env = str(ev.get("api_key_env") or "OPENAI_API_KEY")
    base_url_env = str(ev.get("base_url_env") or "OPENAI_BASE_URL")
    model_env = str(ev.get("model_env") or "OPENAI_MODEL")
    # Credential presence is validated before any output labels/checkpoints are created.
    _require_env(api_key_env); base_url = _require_env(base_url_env); model = _require_env(model_env)
    return {"api_key_env": api_key_env, "base_url_env": base_url_env, "model_env": model_env, "base_url": base_url, "model": model}


def cache_identity(*, agent_type: str, scenario_hash: str, transition_pair_hash: str, prompt_version: str, schema_version: str, evaluator_model: str, evaluator_parameters: dict[str, Any]) -> str:
    payload = {
        "agent_type": agent_type,
        "scenario_hash": scenario_hash,
        "transition_pair_hash": transition_pair_hash,
        "prompt_version": prompt_version,
        "schema_version": schema_version,
        "evaluator_model": evaluator_model,
        "evaluator_parameters": evaluator_parameters,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_structured_response(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"non-JSON evaluator output: {exc.msg}") from exc
    if data.get("preferred") not in {"A", "B", "equal"}:
        raise ValueError("unknown preferred value")
    conf = float(data.get("confidence"))
    if not (0.0 <= conf <= 1.0):
        raise ValueError("nonfinite or out-of-range confidence")
    if not isinstance(data.get("criteria"), dict) or not isinstance(data.get("reason"), str) or not data["reason"].strip():
        raise ValueError("missing required structured fields")
    forbidden = ("RLAIF", "MAPPO", "PPO", "policy", "algorithm")
    if any(word.lower() in data["reason"].lower() for word in forbidden):
        raise ValueError("response refers to hidden method names")
    return data


def _write_reward_train_config(path: Path, base: dict[str, Any], agent: str) -> None:
    cfg = json.loads(json.dumps(base))
    cfg.setdefault("validation", {})["require_all_event_types_present"] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _validate_checkpoint(agent: str, path: Path, digest: str) -> None:
    RuntimeAgentRewardModel.from_checkpoint(
        path,
        expected_agent_type=agent,
        expected_event_types=sorted(REQUIRED_EVENT_COVERAGE[agent]),
        expected_checkpoint_hash=digest,
        formal_mode=True,
    )


def _write_runtime_config(path: Path, manifest: dict[str, Any]) -> None:
    agents = {}
    for agent in AGENT_TYPES:
        rec = manifest["agents"][agent]
        agents[agent] = {
            "enabled": True,
            "checkpoint": rec["checkpoint_path"],
            "checkpoint_hash": rec["checkpoint_hash"],
            "lambda": 1.0,
            "reward_clip": 2.0,
            "supported_event_types": rec["supported_event_types"],
        }
    cfg = {
        "run_classification": "formal",
        "publication_eligible": False,
        "method_id": "mappo_rlaif_all",
        "rlaif": {"enabled": True, "scope": "all", "fallback_to_env_reward": False, "fail_on_invalid_reward_model": True, "agents": agents},
    }
    text = yaml.safe_dump(cfg, sort_keys=False)
    if any(p in text for p in PLACEHOLDERS):
        raise RuntimeError("unresolved placeholder in generated full-RLAIF runtime config")
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)


def prepare(config_path: Path, output_root: Path, *, resume: bool) -> dict[str, Any]:
    cfg = _load_yaml(config_path)
    if cfg.get("run_classification") != "formal":
        raise RuntimeError("formal RLAIF artifact preparation requires run_classification=formal")
    evaluator = verify_evaluator_config(cfg)
    manifest: dict[str, Any] = {"status": "incomplete", "code_commit": _git_sha(), "evaluator_model": evaluator["model"], "prompt_version": cfg["evaluator"]["prompt_version"], "agents": {}}
    model_cfg_base = cfg["reward_model"]["config_template"]
    for agent in AGENT_TYPES:
        acfg = cfg["agents"][agent]
        pref = Path(acfg["output_preferences"])
        if not pref.is_file():
            raise RuntimeError(f"missing formal preference data for {agent}: {pref}")
        rm_cfg = output_root / "reward_model_configs" / f"reward_{agent}.yaml"
        _write_reward_train_config(rm_cfg, model_cfg_base, agent)
        ckpt = Path(acfg["reward_checkpoint"])
        if not (resume and ckpt.is_file()):
            rc = train_reward_main(["--preferences", str(pref), "--config", str(rm_cfg), "--agent", agent, "--output", str(ckpt), "--device", "cpu"])
            if rc != 0:
                raise RuntimeError(f"reward-model training failed for {agent} with exit code {rc}")
        digest = sha256_file(ckpt)
        _validate_checkpoint(agent, ckpt, digest)
        manifest["agents"][agent] = {
            "preference_path": str(pref),
            "preference_hash": _hash_json(pref),
            "checkpoint_path": str(ckpt),
            "checkpoint_hash": digest,
            "supported_event_types": acfg["supported_event_types"],
        }
    manifest["status"] = "FORMAL_FOUR_AGENT_REWARD_MODELS_VALIDATED"
    out_manifest = Path(cfg["outputs"]["manifest"]); out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write_runtime_config(Path(cfg["outputs"]["full_rlaif_runtime_config"]), manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, default=Path("results/formal/rlaif"))
    ap.add_argument("--resume", action="store_true")
    ns = ap.parse_args(argv)
    try:
        print(json.dumps(prepare(ns.config, ns.output_root, resume=ns.resume), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"formal RLAIF artifact preparation failed: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
