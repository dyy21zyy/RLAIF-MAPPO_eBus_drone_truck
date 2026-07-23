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
from experiments.generate_formal_multiagent_preferences import generate as generate_preferences
from training.config_resolver import resolve_mappo_training_config
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


def cache_identity(*, agent_type: str, scenario_hash: str, transition_pair_hash: str | None = None, event_type: str | None = None, decision_state_hash: str | None = None, candidate_pair_hash: str | None = None, prompt_version: str, schema_version: str, evaluator_model: str, evaluator_parameters: dict[str, Any]) -> str:
    payload = {
        "agent_type": agent_type,
        "event_type": event_type,
        "scenario_hash": scenario_hash,
        "decision_state_hash": decision_state_hash,
        "candidate_pair_hash": candidate_pair_hash or transition_pair_hash,
        "prompt_version": prompt_version,
        "schema_version": schema_version,
        "evaluator_model": evaluator_model,
        "temperature": evaluator_parameters.get("temperature"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_structured_response(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"non-JSON evaluator output: {exc.msg}") from exc
    if data.get("preferred") not in {"A", "B", "equal"}:
        raise ValueError("unknown preferred value")
    if "confidence" not in data:
        raise ValueError("missing confidence")
    conf = float(data.get("confidence"))
    if not (0.0 <= conf <= 1.0):
        raise ValueError("nonfinite or out-of-range confidence")
    if not isinstance(data.get("criteria"), dict) or not isinstance(data.get("reason"), str) or not data["reason"].strip():
        raise ValueError("missing required structured fields")
    forbidden = ("RLAIF", "MAPPO", "PPO", "policy", "algorithm")
    if any(word.lower() in data["reason"].lower() for word in forbidden):
        raise ValueError("response refers to hidden method names")
    return data



def _validate_preference_file(agent: str, path: Path, target: int) -> dict[str, Any]:
    from rlaif.preference_dataset import read_jsonl
    from rlaif.reward_model_dataset import build_reward_pair_dataset
    rows = read_jsonl(path)
    ds = build_reward_pair_dataset(rows, agent_type=agent, formal_mode=True, require_bus_event_coverage=(agent == "bus"))
    if len(ds.examples) < int(target):
        raise RuntimeError(f"{agent} usable-label target not met: {len(ds.examples)} < {target}")
    splits = {}
    for r in rows:
        splits.setdefault(r.get("dataset_split"), set()).add((r.get("scenario_id"), r.get("scenario_hash")))
    if set(splits) - {"train", "validation", "test"} or {"train", "validation", "test"} - set(splits):
        raise RuntimeError(f"{agent} missing train/validation/test preference splits")
    if splits["train"] & splits["validation"] or splits["train"] & splits["test"] or splits["validation"] & splits["test"]:
        raise RuntimeError(f"{agent} preference split leakage detected")
    return {"rows": len(rows), "usable_binary": len(ds.examples), "counts_by_event": dict(ds.report.counts_by_event)}

def _inject_artifacts(template_path: Path, output_path: Path, manifest: dict[str, Any], scope_agents: tuple[str, ...], cfg: dict[str, Any]) -> None:
    tpl = _load_yaml(template_path)
    train_manifest = Path(cfg["scenario_bank"]["final_train_manifest"])
    bank_hash = sha256_file(train_manifest) if train_manifest.exists() else manifest.get("scenario_bank_hash", "")
    scale = Path(tpl.get("reward", {}).get("scale_artifact", "results/formal/reward_scales/final_reward_reference_scales.json"))
    tpl.setdefault("env", {})["scenario_bank_manifest"] = str(train_manifest)
    tpl["env"]["expected_split"] = "train"
    tpl["env"]["expected_bank_hash"] = bank_hash
    tpl.setdefault("reward", {})["expected_training_scenario_bank_hash"] = bank_hash
    if scale.exists():
        tpl["reward"]["scale_artifact_hash"] = _hash_json(scale)
    else:
        tpl["reward"]["scale_artifact_hash"] = manifest.get("reward_scale_hash", bank_hash)
    tpl.setdefault("rlaif", {})["fallback_to_env_reward"] = False
    tpl["rlaif"]["fail_on_invalid_reward_model"] = True
    for agent in AGENT_TYPES:
        arec = tpl["rlaif"].setdefault("agents", {}).setdefault(agent, {})
        arec["enabled"] = agent in scope_agents
        if agent in scope_agents:
            rec = manifest["agents"][agent]
            arec["checkpoint"] = rec["checkpoint_path"]
            arec["checkpoint_hash"] = rec["checkpoint_hash"]
    tpl.setdefault("output", {})["output_root"] = str(output_path.parent.parent / output_path.stem)
    resolved = resolve_mappo_training_config(tpl)
    text = yaml.safe_dump(resolved, sort_keys=False)
    if any(p in text for p in PLACEHOLDERS):
        raise RuntimeError(f"unresolved placeholder in {output_path}")
    for section in ("run_classification","mode","env","training","networks","reward","rlaif","output"):
        if section not in resolved: raise RuntimeError(f"resolved config missing {section}")
    output_path.parent.mkdir(parents=True, exist_ok=True); output_path.write_text(text)

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
    train_manifest = Path(cfg["scenario_bank"]["final_train_manifest"])
    if not train_manifest.is_file():
        raise RuntimeError(f"missing formal train bank manifest: {train_manifest}")
    evaluator = verify_evaluator_config(cfg)
    pref_manifest = output_root / "preference_manifest.json"
    need_generate = not pref_manifest.is_file()
    if not need_generate:
        try:
            pm = json.loads(pref_manifest.read_text())
            for agent in AGENT_TYPES:
                _validate_preference_file(agent, Path(cfg["agents"][agent]["output_preferences"]), cfg["agents"][agent]["target_valid_pair_count"])
                if agent not in pm.get("agents", {}): raise RuntimeError("stale manifest")
        except Exception:
            need_generate = True
    if need_generate:
        generate_preferences(config_path, output_root, resume=resume)
    manifest: dict[str, Any] = {"status": "incomplete", "code_commit": _git_sha(), "evaluator_model": evaluator["model"], "prompt_version": cfg["evaluator"]["prompt_version"], "scenario_bank_hash": sha256_file(train_manifest), "agents": {}}
    model_cfg_base = cfg["reward_model"]["config_template"]
    for agent in AGENT_TYPES:
        acfg = cfg["agents"][agent]
        pref = Path(acfg["output_preferences"])
        pref_report = _validate_preference_file(agent, pref, acfg["target_valid_pair_count"])
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
            "preference_validation": pref_report,
            "checkpoint_path": str(ckpt),
            "checkpoint_hash": digest,
            "supported_event_types": acfg["supported_event_types"],
        }
    manifest["status"] = "FORMAL_FOUR_AGENT_REWARD_MODELS_VALIDATED"
    out_manifest = Path(cfg["outputs"]["manifest"]); out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    config_root = output_root.parent / "configs"
    _inject_artifacts(Path("configs/paper/train_mappo_rlaif_assignment.yaml"), config_root / "mappo_rlaif_assignment.yaml", manifest, ("assignment",), cfg)
    _inject_artifacts(Path("configs/paper/train_mappo_rlaif_all.yaml"), config_root / "mappo_rlaif_all.yaml", manifest, tuple(AGENT_TYPES), cfg)
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
