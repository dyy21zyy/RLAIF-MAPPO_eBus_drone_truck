"""Artifact-aware, fail-closed orchestration for the hard-locker formal rerun.

The default invocation is deliberately a dry plan.  ``--execute`` is required for
any subprocess and each phase is revalidated before its content-addressed marker
is written.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

import yaml

TOKENS = ("REPLACE_WITH_", "PLACEHOLDER", "TODO_HASH")
SCOPE = ("assignment",)
SCHEMA_VERSION = 4
MARKER_SCHEMA_VERSION = 1
EXPERIMENT_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ArtifactRecord:
    """A validated, content-addressed artifact (validation stays in validators)."""
    path: str
    hash: str
    validation_status: str = "validated"
    schema_version: int | str | None = None
    lineage_hashes: dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentContext:
    repository_commit: str
    run_root: Path
    selected_agents: tuple[str, ...] = SCOPE
    scenario_banks: dict[str, dict[str, Any]] = field(default_factory=dict)
    preference_artifact: dict[str, Any] | None = None
    reward_model_artifact: dict[str, Any] | None = None
    reward_scale_artifact: dict[str, Any] | None = None
    resolved_training_configs: dict[str, ArtifactRecord] = field(default_factory=dict)
    policy_checkpoints: dict[str, dict[int, dict[str, Any]]] = field(default_factory=dict)
    resolved_benchmark_config: ArtifactRecord | None = None
    final_freeze_artifact: ArtifactRecord | None = None
    device: str = "cuda"

    # Transitional aliases keep validators simple and old callers compatible.
    @property
    def banks(self): return self.scenario_banks
    @property
    def preference(self): return self.preference_artifact
    @property
    def reward_model(self): return self.reward_model_artifact
    @property
    def scale(self): return self.reward_scale_artifact
    @property
    def checkpoints(self): return self.policy_checkpoints


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assignment_preference_path(output_root: Path) -> Path:
    """Path promised by generate_formal_multiagent_preferences' output contract."""
    return output_root / "preferences" / "preference_assignment.jsonl"


def find_placeholders(value: Any, path: str = "$") -> list[str]:
    """Recursively locate unresolved tokens in executable config values."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(find_placeholders(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_placeholders(child, f"{path}[{index}]"))
    elif isinstance(value, str) and any(token in value.upper() for token in TOKENS):
        found.append(path)
    return found


def assert_no_placeholders(value: Any, *, label: str = "formal executable config") -> None:
    found = find_placeholders(value)
    if found:
        raise ValueError(f"{label} contains unresolved placeholders at {', '.join(found)}")


def _load_data(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def _dump_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def validate_assignment_preferences(output_root: Path) -> dict[str, Any]:
    path = assignment_preference_path(output_root)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"assignment preference JSONL is missing, not a regular file, or empty: {path}")
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL record {line_no}") from exc
    if not rows:
        raise ValueError("assignment preference JSONL contains no records")
    if any(row.get("agent_type") != "assignment" for row in rows):
        raise ValueError("assignment preference file contains a non-assignment record")
    for key in ("observation_schema_version", "candidate_schema_version"):
        if any(row.get(key) != SCHEMA_VERSION for row in rows):
            raise ValueError(f"assignment preference {key} must be {SCHEMA_VERSION}")
    splits = {row.get("dataset_split") for row in rows}
    if splits != {"train", "validation", "test"}:
        raise ValueError(f"assignment preference split coverage is incomplete: {sorted(map(str, splits))}")
    manifest_path = output_root / "preference_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if not isinstance(manifest, dict):
        raise ValueError("preference manifest is missing")
    if manifest.get("selected_agents") != ["assignment"]:
        raise ValueError("preference manifest is not assignment-only")
    entry = manifest.get("agents", {}).get("assignment", {})
    if Path(entry.get("path", "")) != path or entry.get("hash") != sha256_file(path):
        raise ValueError("preference manifest path/hash does not match assignment JSONL")
    counts = entry.get("counts_by_split", {})
    if any(int(counts.get(split, 0)) <= 0 for split in ("train", "validation", "test")):
        raise ValueError("preference manifest split coverage is incomplete")
    return {"path": str(path), "hash": sha256_file(path), "manifest_path": str(manifest_path),
            "manifest_hash": sha256_file(manifest_path), "record_count": len(rows), "selected_agents": ["assignment"]}


def validate_scenario_manifest(path: Path, expected_split: str | None = None) -> dict[str, Any]:
    data = _load_data(path)
    split = data.get("split") or data.get("scenario_split")
    if expected_split and split != expected_split:
        raise ValueError(f"scenario bank {path} has split {split!r}, expected {expected_split!r}")
    bank_hash = data.get("bank_hash") or data.get("scenario_bank_hash")
    if not isinstance(bank_hash, str) or not bank_hash:
        raise ValueError(f"scenario bank hash missing in {path}")
    return {"path": str(path), "bank_hash": bank_hash, "manifest_hash": sha256_file(path), "split": split}


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"checkpoint missing: {path}")
    try:
        import torch
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return data


def validate_reward_checkpoint(path: Path, preference: dict[str, Any]) -> dict[str, Any]:
    data = _load_checkpoint(path)
    if data.get("validation_status") not in ("pass", "passed", "validated"):
        raise ValueError("assignment reward checkpoint did not pass validation")
    if data.get("agent_type") != "assignment":
        raise ValueError("reward checkpoint is not assignment-only")
    for key in ("observation_schema_version", "candidate_schema_version"):
        if data.get(key) != SCHEMA_VERSION:
            raise ValueError(f"reward checkpoint {key} must be {SCHEMA_VERSION}")
    if data.get("preference_file_hash") != preference["hash"]:
        raise ValueError("reward checkpoint preference lineage mismatch")
    return {"path": str(path), "hash": sha256_file(path), "validation_status": data["validation_status"],
            "observation_schema_version": SCHEMA_VERSION, "candidate_schema_version": SCHEMA_VERSION,
            "preference_file_hash": preference["hash"]}


def validate_reward_scale(path: Path, train_bank_hash: str) -> dict[str, Any]:
    data = _load_data(path)
    if data.get("validation_status") not in ("pass", "passed", "validated"):
        raise ValueError("reward scale artifact did not pass validation")
    lineage = data.get("training_scenario_bank_hash") or data.get("training_bank_hash") or data.get("scenario_bank_hash")
    if lineage != train_bank_hash:
        raise ValueError("reward scale training-bank lineage mismatch")
    components = data.get("components") or data.get("scales") or {}
    locker = components.get("locker_overflow", {})
    if isinstance(locker, (int, float)):
        raise ValueError("locker_overflow scale must explicitly document structural zero and denominator")
    denominator = locker.get("denominator") or locker.get("reference_scale") or locker.get("scale")
    if locker.get("structural_zero") is not True or not isinstance(denominator, (int, float)) or denominator <= 0:
        raise ValueError("locker_overflow must be structural zero with an explicit positive denominator")
    from envs.reward_scales import canonical_payload_hash
    canonical = canonical_payload_hash(data)
    if data.get("artifact_hash") != canonical:
        raise ValueError("reward scale canonical artifact_hash mismatch")
    return {"path": str(path), "reward_scale_artifact_hash": canonical, "hash": canonical, "reward_scale_file_sha256": sha256_file(path), "training_bank_hash": train_bank_hash,
            "validation_status": data["validation_status"]}


def materialize_initial_configs(root: Path, device: str = "cuda") -> Path:
    """Materialize only the reward-model config, whose inputs are already known."""
    destination = root / "resolved_configs" / "train_reward_assignment.yaml"
    config = _load_data(Path("configs/paper/train_reward_assignment.yaml"))
    config.setdefault("training", {}).update({"device": device, "require_cuda": device != "cpu"})
    _dump_yaml(destination, config)
    return destination

def gpu_readiness(path: Path, requested: str, *, require_cuda: bool = True) -> dict[str, Any]:
    import torch
    from training.device import resolve_torch_device
    device=resolve_torch_device(requested, require_cuda=require_cuda)
    index=device.index if device.type == "cuda" else None; props=torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    smoke_status="not_required_cpu"
    if device.type == "cuda":
        a=torch.ones((8,8),device=device); smoke=float((a@a).sum().item()); smoke_status="passed"
    record={"marker":"FORMAL_CUDA_READY" if device.type == "cuda" else None,"requested_device":requested,"resolved_device":str(device),"require_cuda":require_cuda,"torch_version":torch.__version__,"cuda_runtime_version":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"device_count":torch.cuda.device_count(),"device_index":index,"device_name":props.name if props else None,"compute_capability":list(torch.cuda.get_device_capability(device)) if props else None,"total_memory_bytes":props.total_memory if props else None,"smoke_status":smoke_status}
    path.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n"); return record


def resolve_training_configs(root: Path, banks: dict[str, dict[str, Any]], reward_model: dict[str, Any], scale: dict[str, Any], device: str = "cuda") -> tuple[Path, Path]:
    resolved = root / "resolved_configs"
    outputs = []
    for method, template in (("mappo_env", "train_mappo_env.yaml"), ("mappo_rlaif_assignment", "train_mappo_rlaif_assignment.yaml")):
        cfg = _load_data(Path("configs/paper") / template)
        cfg.setdefault("training", {}).update({"device":device,"require_cuda":device != "cpu"})
        cfg["scenario_bank"] = {"manifest": banks["train"]["path"], "bank_hash": banks["train"]["bank_hash"]}
        cfg.setdefault("env", {}).update({"scenario_bank_manifest": banks["train"]["path"], "expected_split":"train", "expected_bank_hash":banks["train"]["bank_hash"], "scenario_sampling_mode":"shuffled_cycle"})
        cfg["reward"]["scale_artifact"] = scale["path"]
        cfg["reward"]["reward_scale_artifact_hash"] = scale["hash"]
        cfg["reward"]["expected_training_scenario_bank_hash"] = banks["train"]["bank_hash"]
        cfg["output"]["output_root"] = str(root / method)
        if method == "mappo_env":
            if cfg.get("rlaif", {}).get("enabled"):
                raise ValueError("MAPPO-Env must not enable learned reward")
        else:
            rlaif = cfg["rlaif"]
            if rlaif.get("scope") != "assignment" or rlaif.get("fallback_to_env_reward") is not False:
                raise ValueError("RLAIF-MAPPO must be assignment-only with fallback disabled")
            rlaif["agents"]["assignment"]["checkpoint"] = reward_model["path"]
            rlaif["agents"]["assignment"]["checkpoint_hash"] = reward_model["hash"]
            rlaif["reward_checkpoint_paths"] = {"assignment": reward_model["path"]}
            rlaif["reward_checkpoint_hashes"] = {"assignment": reward_model["hash"]}
            rlaif["reward_model_schema_versions"] = {"assignment": SCHEMA_VERSION}
            if any(v.get("enabled", False) for k, v in rlaif["agents"].items() if k != "assignment"):
                raise ValueError("non-assignment learned rewards must be disabled")
        assert_no_placeholders(cfg, label=template)
        out = resolved / template
        _dump_yaml(out, cfg); outputs.append(out)
    return outputs[0], outputs[1]


def validate_policy_checkpoint(path: Path, *, method: str, seed: int, commit: str,
                               banks: dict[str, dict[str, Any]], reward_model: dict[str, Any], scale: dict[str, Any]) -> dict[str, Any]:
    ck = _load_checkpoint(path)
    if ck.get("checkpoint_schema_version") != SCHEMA_VERSION:
        raise ValueError("policy checkpoint schema version must be 4")
    if int(ck.get("optimizer_updates", ck.get("optimization_steps", 0))) <= 0:
        raise ValueError("policy checkpoint has no optimizer updates")
    if ck.get("code_commit") != commit:
        raise ValueError("policy checkpoint commit lineage mismatch")
    if ck.get("training_seed", ck.get("seed")) != seed:
        raise ValueError("policy checkpoint seed mismatch")
    if ck.get("training_scenario_bank_hash") != banks["train"]["bank_hash"]:
        raise ValueError("policy checkpoint scenario lineage mismatch")
    if ck.get("reward_scale_artifact_hash") != scale["hash"]:
        raise ValueError("policy checkpoint reward-scale lineage mismatch")
    if method == "mappo_rlaif_assignment":
        hashes = ck.get("reward_checkpoint_hashes", {})
        if hashes.get("assignment") != reward_model["hash"]:
            raise ValueError("policy checkpoint reward-model lineage mismatch")
    actor = ck.get("actor_specs")
    critic = ck.get("critic_spec") or ck.get("critic_state_dict")
    if not actor or not critic:
        raise ValueError("policy checkpoint actor/critic feature metadata missing")
    return {"path": str(path), "hash": sha256_file(path), "method": method, "seed": seed}


def discover_policy_checkpoints(root: Path, commit: str, banks: dict[str, dict[str, Any]], reward_model: dict[str, Any], scale: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    found: dict[str, dict[int, dict[str, Any]]] = {"mappo_env":{},"mappo_rlaif_assignment":{}}
    seen: set[tuple[str,int]] = set()
    for method in found:
        for seed in (1,2,3):
            manifest=root/method/f"seed_{seed}"/"training_run_manifest.json"
            if not manifest.is_file(): raise ValueError(f"formal training manifest missing: {manifest}")
            try: data=json.loads(manifest.read_text())
            except Exception as exc: raise ValueError(f"invalid training manifest JSON: {manifest}") from exc
            required=("method_id","training_seed","status","code_commit","resolved_config_path","resolved_config_hash","checkpoint_path","checkpoint_file_sha256","checkpoint_schema_version","optimizer_updates","scenario_bank_path","scenario_bank_hash","reward_scale_path","reward_scale_artifact_hash","reward_model_path","reward_model_hash","start_time","completion_time")
            missing=[key for key in required if key not in data]
            if missing: raise ValueError(f"training manifest missing fields {missing}: {manifest}")
            key=(data["method_id"],int(data["training_seed"]))
            if key in seen: raise ValueError(f"duplicate method/seed manifest: {key}")
            seen.add(key)
            if data["method_id"] != method: raise ValueError(f"training manifest method mismatch: {manifest}")
            if int(data["training_seed"]) != seed: raise ValueError(f"training manifest seed mismatch: {manifest}")
            if data["status"] != "complete" or data["code_commit"] != commit: raise ValueError(f"training manifest status/commit mismatch: {manifest}")
            if data["scenario_bank_hash"] != banks["train"]["bank_hash"]: raise ValueError("training manifest scenario-bank hash mismatch")
            if data["reward_scale_artifact_hash"] != scale["hash"]: raise ValueError("training manifest reward-scale hash mismatch")
            if method == "mappo_env" and (data["reward_model_path"] is not None or data["reward_model_hash"] is not None): raise ValueError("MAPPO-Env manifest must not contain reward-model lineage")
            if method == "mappo_rlaif_assignment" and (data["reward_model_path"] != reward_model["path"] or data["reward_model_hash"] != reward_model["hash"]): raise ValueError("RLAIF manifest assignment reward-model lineage mismatch")
            candidate=Path(data["checkpoint_path"])
            if not candidate.is_file() or data["checkpoint_file_sha256"] != sha256_file(candidate): raise ValueError(f"training manifest checkpoint hash mismatch: {manifest}")
            record=validate_policy_checkpoint(candidate,method=method,seed=seed,commit=commit,banks=banks,reward_model=reward_model,scale=scale)
            record["manifest_path"]=str(manifest); record["manifest_hash"]=sha256_file(manifest); found[method][seed]=record
    if len(seen) != 6: raise ValueError(f"exactly six unique formal training manifests required, found {len(seen)}")
    return found


def resolve_benchmark_config(root: Path, banks: dict[str, dict[str, Any]], reward_model: dict[str, Any], checkpoints: dict[str, dict[int, dict[str, Any]]]) -> Path:
    cfg = _load_data(Path("configs/paper/benchmark.yaml"))
    cfg["scenario_bank"]["manifest"] = banks["test"]["path"]
    cfg["scenario_bank"]["expected_bank_hash"] = banks["test"]["bank_hash"]
    first_checkpoint = _load_checkpoint(Path(checkpoints["mappo_env"][1]["path"]))
    cfg["reward_scale_artifact_path"] = first_checkpoint.get("reward_scale_artifact_path")
    cfg["reward_scale_artifact_hash"] = first_checkpoint.get("reward_scale_artifact_hash")
    cfg["methods"] = [m for m in cfg["methods"] if m.get("method_id") in checkpoints]
    for method in cfg["methods"]:
        mid = method["method_id"]
        method["policy_checkpoints"] = {str(seed): item["path"] for seed, item in checkpoints[mid].items()}
        method["policy_checkpoint_hashes"] = {str(seed): item["hash"] for seed, item in checkpoints[mid].items()}
        if mid == "mappo_rlaif_assignment":
            method["reward_checkpoints"] = {"assignment": reward_model["path"]}
            method["reward_checkpoint_hashes"] = {"assignment": reward_model["hash"]}
    assert_no_placeholders(cfg, label="benchmark.yaml")
    out = root / "resolved_configs" / "benchmark.yaml"; _dump_yaml(out, cfg)
    return out


def resolve_final_freeze_config(root: Path, banks: dict[str, dict[str, Any]], reward_model: dict[str, Any], scale: dict[str, Any], checkpoints: dict[str, dict[int, dict[str, Any]]]) -> Path:
    cfg = _load_data(Path("configs/paper/final_experiment_freeze.template.yaml"))
    replacements = {
        "REPLACE_WITH_FINAL_TRAIN_BANK_HASH": banks["train"]["bank_hash"],
        "REPLACE_WITH_FINAL_VALIDATION_BANK_HASH": banks["validation"]["bank_hash"],
        "REPLACE_WITH_FINAL_TEST_BANK_HASH": banks["test"]["bank_hash"],
        "REPLACE_WITH_FINAL_REWARD_SCALE_HASH": scale["hash"],
        "REPLACE_WITH_REAL_SCALE_HASH": scale["hash"],
        "REPLACE_WITH_FINAL_ASSIGNMENT_REWARD_HASH": reward_model["hash"],
    }
    def replace(value: Any) -> Any:
        if isinstance(value, dict): return {k: replace(v) for k, v in value.items()}
        if isinstance(value, list): return [replace(v) for v in value]
        if isinstance(value, str):
            for old, new in replacements.items(): value = value.replace(old, new)
            value = value.replace("results/formal/reward_scales/final_reward_reference_scales.json", scale["path"])
            value = value.replace("results/formal/reward_models/reward_assignment.pt", reward_model["path"])
        return value
    cfg = replace(cfg)
    cfg.setdefault("artifact_contract", {})["resolved_policy_checkpoints"] = {
        method: {str(seed): item for seed, item in values.items()} for method, values in checkpoints.items()}
    # The assignment-only formal rerun intentionally removes optional four-agent placeholders.
    methods = cfg.get("rlaif_parameters", {}).get("methods", {})
    methods.pop("mappo_rlaif_all", None)
    assert_no_placeholders(cfg, label="final freeze config")
    out = root / "resolved_configs" / "final_experiment_freeze.yaml"; _dump_yaml(out, cfg)
    return out


@dataclass(frozen=True)
class PhaseSpec:
    phase: int
    name: str
    commands: tuple[tuple[str, ...], ...]
    outputs: tuple[Path, ...]
    inputs: tuple[Path, ...] = field(default_factory=tuple)
    validation: str = "outputs_exist"
    capture: Path | None = None


def build_plan(*, output_root: Path, commit: str, train_manifest: Path | None = None, device: str = "cuda") -> list[PhaseSpec]:
    py, root = sys.executable, output_root
    pref_root=root/"preferences"; pref_file=assignment_preference_path(pref_root)
    reward=root/"reward_models"/"reward_assignment.pt"; scale=root/"reward_scales"/"final_reward_reference_scales.json"
    resolved=root/"resolved_configs"; benchmark=root/"benchmark"; banks=_scenario_banks(); train=train_manifest or banks["train"]
    env_roots=tuple(root/"mappo_env"/f"seed_{s}" for s in (1,2,3)); rlaif_roots=tuple(root/"mappo_rlaif_assignment"/f"seed_{s}" for s in (1,2,3))
    manifests=tuple(r/"training_run_manifest.json" for r in (*env_roots,*rlaif_roots))
    # Every dependency which can change execution is declared and content-addressed.
    return [
      PhaseSpec(0,"initial_artifact_resolution",(),(resolved/"train_reward_assignment.yaml",root/"gpu_readiness.json"),(banks["train"],banks["validation"],banks["test"],Path("configs/paper/train_reward_assignment.yaml"))),
      PhaseSpec(1,"verification",((py,"-m","pytest","-q","tests/test_hard_locker_capacity.py"),),(root/"verification.json",),(Path("tests/test_hard_locker_capacity.py"),)),
      PhaseSpec(2,"assignment_preferences_and_reward_model",((py,"-m","experiments.generate_formal_multiagent_preferences","--config","configs/paper/rlaif_preference_generation.yaml","--output-root",str(pref_root),"--agents","assignment","--cache-dir",str(pref_root/"evaluator_cache")),(py,"-m","experiments.train_multi_agent_reward_models","--preferences",str(pref_file),"--config",str(resolved/"train_reward_assignment.yaml"),"--agent","assignment","--output",str(reward),"--device",device)),(pref_file,pref_root/"preference_manifest.json",reward),(Path("configs/paper/rlaif_preference_generation.yaml"),train,Path("configs/paper/train_reward_assignment.yaml")),"assignment_preference_and_reward"),
      PhaseSpec(3,"reward_scales_and_training_config_resolution",((py,"-m","experiments.estimate_reward_reference_scales","--scenario-bank",str(train),"--config","configs/paper/reward_scale_estimation.yaml","--output",str(scale)),),(scale,resolved/"train_mappo_env.yaml",resolved/"train_mappo_rlaif_assignment.yaml"),(train,Path("configs/paper/reward_scale_estimation.yaml"),reward)),
      PhaseSpec(4,"placeholder_free_pretraining_gate",(),(resolved/"train_mappo_env.yaml",resolved/"train_mappo_rlaif_assignment.yaml"),(resolved/"train_mappo_env.yaml",resolved/"train_mappo_rlaif_assignment.yaml",scale,reward)),
      PhaseSpec(5,"mappo_env_training",tuple((py,"-m","experiments.train_mappo_async","--config",str(resolved/"train_mappo_env.yaml"),"--seed",str(seed),"--output-root",str(env_roots[seed-1])) for seed in (1,2,3)),env_roots,(resolved/"train_mappo_env.yaml",train,scale)),
      PhaseSpec(6,"rlaif_mappo_training",tuple((py,"-m","experiments.train_mappo_async","--config",str(resolved/"train_mappo_rlaif_assignment.yaml"),"--seed",str(seed),"--output-root",str(rlaif_roots[seed-1])) for seed in (1,2,3)),rlaif_roots,(resolved/"train_mappo_rlaif_assignment.yaml",train,scale,reward)),
      PhaseSpec(7,"checkpoint_resolution_and_final_freeze",((py,"-m","experiments.validate_formal_experiment_readiness","--config",str(resolved/"benchmark.yaml"),"--strict"),(py,"-m","experiments.freeze_final_experiment_parameters","--config",str(resolved/"final_experiment_freeze.yaml"),"--output",str(root/"final_experiment_freeze.json"))),(resolved/"benchmark.yaml",resolved/"final_experiment_freeze.yaml",root/"final_experiment_freeze.json"),manifests+(banks["test"],reward,scale,Path("configs/paper/benchmark.yaml"),Path("configs/paper/final_experiment_freeze.template.yaml"))),
      PhaseSpec(8,"benchmark_600_rows",((py,"-m","experiments.run_paper_benchmark","--config",str(resolved/"benchmark.yaml"),"--output-root",str(benchmark)),),(benchmark/"episode_results.jsonl",),(resolved/"benchmark.yaml",)+manifests+(banks["test"],reward,scale)),
      PhaseSpec(9,"paired_analysis",((py,"-m","experiments.analyze_hard_locker_paired_results","--input",str(benchmark/"episode_results.jsonl"),"--output-dir",str(root/"paired_analysis")),),(root/"paired_analysis"/"analysis_manifest.json",),(benchmark/"episode_results.jsonl",Path("experiments/analyze_hard_locker_paired_results.py"))),
    ]


def _under(root: Path, path: Path) -> bool:
    try: path.resolve().relative_to(root.resolve()); return True
    except ValueError: return False


def validate_plan(plan: Sequence[PhaseSpec], root: Path) -> None:
    for phase in plan:
        for output in phase.outputs:
            if ".." in output.parts or not _under(root, output):
                raise ValueError(f"phase {phase.phase} output escapes run root: {output}")


def plan_hash(plan: Sequence[PhaseSpec]) -> str:
    return hashlib.sha256(json.dumps([asdict(p) for p in plan],sort_keys=True,default=str).encode()).hexdigest()


def _path_hash(path: Path) -> str:
    if path.is_file(): return sha256_file(path)
    if path.is_dir():
        entries={str(p.relative_to(path)):sha256_file(p) for p in sorted(path.rglob("*")) if p.is_file() and "phase_manifests" not in p.parts}
        return hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()
    return "MISSING"


def phase_artifact_state(phase: PhaseSpec, *, commit: str, plan_digest: str, selected_agents: tuple[str,...] = SCOPE) -> dict[str, Any]:
    records=lambda paths: [{"path":str(p),"sha256":_path_hash(p)} for p in paths]
    return {"marker_schema_version":MARKER_SCHEMA_VERSION,"phase":phase.phase,"phase_name":phase.name,
            "name":phase.name,"repository_commit":commit,"commit":commit,"plan_hash":plan_digest,
            "selected_agent_scope":list(selected_agents),"selected_agents":list(selected_agents),
            "inputs":records(phase.inputs),"outputs":records(phase.outputs),
            "input_hashes":{str(p):_path_hash(p) for p in phase.inputs},
            "output_hashes":{str(p):_path_hash(p) for p in phase.outputs},
            "schema_versions":{"artifact":SCHEMA_VERSION,"marker":MARKER_SCHEMA_VERSION},
            "validation_status":"validated"}


def _configs_placeholder_free(phase: PhaseSpec) -> bool:
    for path in (*phase.inputs,*phase.outputs):
        if path.is_file() and path.suffix.lower() in (".yaml",".yml",".json"):
            try:
                if find_placeholders(_load_data(path)): return False
            except (ValueError, yaml.YAMLError): return False
    return True


def marker_validation_errors(marker: Path, phase: PhaseSpec, *, commit: str, plan_digest: str,
                             selected_agents: tuple[str,...]=SCOPE) -> list[str]:
    if not marker.is_file(): return ["marker missing"]
    try: stored=json.loads(marker.read_text())
    except Exception as exc: return [f"marker corrupted: {exc}"]
    current=phase_artifact_state(phase,commit=commit,plan_digest=plan_digest,selected_agents=selected_agents)
    errors=[]
    for key in ("phase","phase_name","repository_commit","plan_hash","selected_agent_scope","input_hashes","output_hashes","schema_versions","validation_status"):
        if stored.get(key) != current.get(key): errors.append(f"{key} mismatch")
    if "MISSING" in current["input_hashes"].values(): errors.append("declared input missing")
    if "MISSING" in current["output_hashes"].values(): errors.append("declared output missing")
    if not _configs_placeholder_free(phase): errors.append("executable config contains placeholders")
    return errors


def marker_is_valid(marker: Path, phase: PhaseSpec, *, commit: str, plan_digest: str,
                    selected_agents: tuple[str,...]=SCOPE) -> bool:
    return not marker_validation_errors(marker,phase,commit=commit,plan_digest=plan_digest,selected_agents=selected_agents)


def atomic_write_json(path: Path, value: dict[str,Any]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as handle:
            json.dump(value,handle,indent=2,sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp,path)
    except BaseException:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise


def write_phase_marker(marker: Path, phase: PhaseSpec, *, commit: str, plan_digest: str,
                       validator: Callable[[], None] | None = None,
                       selected_agents: tuple[str,...]=SCOPE) -> None:
    if validator: validator()
    state=phase_artifact_state(phase,commit=commit,plan_digest=plan_digest,selected_agents=selected_agents)
    missing=[p for p,h in {**state["input_hashes"],**state["output_hashes"]}.items() if h=="MISSING"]
    if missing: raise ValueError(f"phase artifacts missing: {missing}")
    if not _configs_placeholder_free(phase): raise ValueError("phase executable config contains unresolved placeholders")
    state["completion_timestamp"]=datetime.now(timezone.utc).isoformat()
    atomic_write_json(marker,state)


def update_experiment_artifact_manifest(context: ExperimentContext, completed_phases: Sequence[int]) -> Path:
    root=context.run_root
    def rec(value):
        if not value: return None
        if isinstance(value,ArtifactRecord): return asdict(value)
        return value
    paired=root/"paired_analysis"; benchmark=root/"benchmark"/"episode_results.jsonl"
    value={"manifest_schema_version":EXPERIMENT_MANIFEST_SCHEMA_VERSION,"repository_commit":context.repository_commit,
      "selected_agents":list(context.selected_agents),"scenario_banks":context.scenario_banks,
      "assignment_preference":rec(context.preference_artifact),"assignment_reward_model":rec(context.reward_model_artifact),
      "reward_scale":rec(context.reward_scale_artifact),"policy_checkpoints":context.policy_checkpoints,
      "benchmark_result":{"path":str(benchmark),"hash":_path_hash(benchmark)} if benchmark.is_file() else None,
      "paired_analysis":[{"path":str(p),"hash":sha256_file(p)} for p in sorted(paired.glob("*")) if p.is_file()],
      "final_freeze_artifact":rec(context.final_freeze_artifact),"completed_phases":sorted(set(completed_phases)),
      "updated_at":datetime.now(timezone.utc).isoformat()}
    path=root/"experiment_artifact_manifest.json"; atomic_write_json(path,value); return path


def _git(*args: str) -> str: return subprocess.check_output(["git",*args],text=True).strip()


def _scenario_banks() -> dict[str, Path]:
    candidates={"train":(Path("results/formal/scenario_banks/train/manifest.json"),Path("results/formal/scenarios/train/scenario_bank_manifest.json")),"validation":(Path("results/formal/scenario_banks/validation/manifest.json"),Path("results/formal/scenarios/validation/scenario_bank_manifest.json")),"test":(Path("results/formal/scenario_banks/test/manifest.json"),Path("data/scenarios/test/scenario_bank_manifest.json"))}
    return {split:next((p for p in paths if p.is_file()),paths[0]) for split,paths in candidates.items()}


def restore_phase_0_context(context: ExperimentContext, bank_paths: dict[str,Path]|None=None) -> None:
    paths=bank_paths or _scenario_banks()
    context.scenario_banks={s:validate_scenario_manifest(p,s) for s,p in paths.items()}
    if context.selected_agents != SCOPE: raise ValueError("formal RLAIF scope must be assignment-only")


def restore_phase_2_context(context: ExperimentContext) -> None:
    context.preference_artifact=validate_assignment_preferences(context.run_root/"preferences")
    context.reward_model_artifact=validate_reward_checkpoint(context.run_root/"reward_models"/"reward_assignment.pt",context.preference_artifact)


def restore_phase_3_context(context: ExperimentContext) -> None:
    if not context.scenario_banks: restore_phase_0_context(context)
    if not context.reward_model_artifact: restore_phase_2_context(context)
    context.reward_scale_artifact=validate_reward_scale(context.run_root/"reward_scales"/"final_reward_reference_scales.json",context.scenario_banks["train"]["bank_hash"])
    context.resolved_training_configs={}
    for method,name in (("mappo_env","train_mappo_env.yaml"),("mappo_rlaif_assignment","train_mappo_rlaif_assignment.yaml")):
        path=context.run_root/"resolved_configs"/name; cfg=_load_data(path); assert_no_placeholders(cfg,label=name)
        if cfg.get("env",{}).get("expected_bank_hash") != context.scenario_banks["train"]["bank_hash"]: raise ValueError(f"{name} scenario lineage mismatch")
        if cfg.get("reward",{}).get("reward_scale_artifact_hash") != context.reward_scale_artifact["hash"]: raise ValueError(f"{name} scale lineage mismatch")
        context.resolved_training_configs[method]=ArtifactRecord(str(path),sha256_file(path),schema_version=SCHEMA_VERSION)


def restore_training_context(context: ExperimentContext) -> None:
    if not context.reward_scale_artifact: restore_phase_3_context(context)
    context.policy_checkpoints=discover_policy_checkpoints(context.run_root,context.repository_commit,context.scenario_banks,context.reward_model_artifact,context.reward_scale_artifact)


def restore_phase_7_context(context: ExperimentContext, *, regenerate: bool=False) -> None:
    if not context.policy_checkpoints: restore_training_context(context)
    benchmark=context.run_root/"resolved_configs"/"benchmark.yaml"; freeze=context.run_root/"resolved_configs"/"final_experiment_freeze.yaml"
    # Regeneration is deterministic and prevents trusting stale referenced hashes.
    if regenerate or not benchmark.is_file(): resolve_benchmark_config(context.run_root,context.scenario_banks,context.reward_model_artifact,context.policy_checkpoints)
    else:
        cfg=_load_data(benchmark); assert_no_placeholders(cfg,label=str(benchmark))
        expected={m:{str(s):x["hash"] for s,x in seeds.items()} for m,seeds in context.policy_checkpoints.items()}
        actual={m["method_id"]:m.get("policy_checkpoint_hashes",{}) for m in cfg.get("methods",[]) if m.get("method_id") in expected}
        if actual != expected: raise ValueError("benchmark checkpoint hashes are stale")
    if regenerate or not freeze.is_file(): resolve_final_freeze_config(context.run_root,context.scenario_banks,context.reward_model_artifact,context.reward_scale_artifact,context.policy_checkpoints)
    else: assert_no_placeholders(_load_data(freeze),label=str(freeze))
    context.resolved_benchmark_config=ArtifactRecord(str(benchmark),sha256_file(benchmark),schema_version=SCHEMA_VERSION)
    context.final_freeze_artifact=ArtifactRecord(str(freeze),sha256_file(freeze),schema_version=SCHEMA_VERSION)


def restore_context_through_phase(root: Path, phase: int, context: ExperimentContext, commit: str|None=None) -> None:
    if phase>=0 and not context.scenario_banks: restore_phase_0_context(context)
    if phase>=2: restore_phase_2_context(context)
    if phase>=3: restore_phase_3_context(context)
    if phase>=6: restore_training_context(context)
    if phase>=7: restore_phase_7_context(context)


def record_failure(root: Path, phase: PhaseSpec, command: Sequence[str]|None, exc: BaseException,
                   stdout_path: Path|None=None, stderr_path: Path|None=None) -> Path:
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report={"phase":phase.phase,"phase_name":phase.name,"command":list(command or ()),"exit_code":getattr(exc,"returncode",None),"stdout_path":str(stdout_path) if stdout_path else None,"stderr_path":str(stderr_path) if stderr_path else None,"timestamp":datetime.now(timezone.utc).isoformat(),"exception_type":type(exc).__name__,"exception":str(exc),"known_output_paths":[str(p) for p in phase.outputs],"partial_artifacts":{str(p):p.exists() for p in phase.outputs}}
    path=root/"failures"/f"phase_{phase.phase}_{stamp}.json"; atomic_write_json(path,report); return path


def _run_command(command: Sequence[str],phase: PhaseSpec,root: Path,index: int) -> None:
    logs=root/"logs"; logs.mkdir(parents=True,exist_ok=True); stdout=logs/f"phase_{phase.phase}_{index}.stdout.log"; stderr=logs/f"phase_{phase.phase}_{index}.stderr.log"
    try:
        with stdout.open("a") as out, stderr.open("a") as err: subprocess.run(command,check=True,stdout=out,stderr=err)
    except BaseException as exc:
        record_failure(root,phase,command,exc,stdout,stderr); raise


def _run_execute_phase(phase: PhaseSpec, context: ExperimentContext, digest: str) -> None:
    root=context.run_root
    try:
        if phase.phase==0: materialize_initial_configs(root,context.device); gpu_readiness(root/"gpu_readiness.json",context.device,require_cuda=context.device != "cpu"); restore_phase_0_context(context)
        elif phase.phase==1: _run_command(phase.commands[0],phase,root,0); (root/"verification.json").write_text(json.dumps({"status":"pass"})+"\n")
        elif phase.phase==2:
            _run_command(phase.commands[0],phase,root,0); context.preference_artifact=validate_assignment_preferences(root/"preferences"); _run_command(phase.commands[1],phase,root,1); context.reward_model_artifact=validate_reward_checkpoint(root/"reward_models"/"reward_assignment.pt",context.preference_artifact)
        elif phase.phase==3:
            _run_command(phase.commands[0],phase,root,0); context.reward_scale_artifact=validate_reward_scale(root/"reward_scales"/"final_reward_reference_scales.json",context.scenario_banks["train"]["bank_hash"]); resolve_training_configs(root,context.scenario_banks,context.reward_model_artifact,context.reward_scale_artifact,context.device); restore_phase_3_context(context)
        elif phase.phase==4:
            for path in phase.outputs: assert_no_placeholders(_load_data(path),label=str(path))
        elif phase.phase in (5,6,8,9):
            for i,command in enumerate(phase.commands): _run_command(command,phase,root,i)
        elif phase.phase==7:
            restore_training_context(context); restore_phase_7_context(context,regenerate=True)
            for i,command in enumerate(phase.commands): _run_command(command,phase,root,i)
        write_phase_marker(root/"phase_manifests"/f"phase_{phase.phase}.json",phase,commit=context.repository_commit,plan_digest=digest)
    except BaseException as exc:
        failures=root/"failures"
        if not failures.exists() or not any(failures.glob(f"phase_{phase.phase}_*.json")): record_failure(root,phase,None,exc)
        raise


def invalidate_markers_from(root: Path, phase: int, maximum: int=9) -> list[int]:
    if phase not in range(maximum+1): raise ValueError(f"invalid phase number: {phase}")
    invalidated=[]
    for number in range(phase,maximum+1):
        marker=root/"phase_manifests"/f"phase_{number}.json"
        if marker.exists(): marker.unlink(); invalidated.append(number)
    return invalidated


def inspect_resume(root: Path, plan: Sequence[PhaseSpec], *, commit: str, digest: str) -> dict[str,Any]:
    valid=[]; invalid={}; missing=[]; changed=[]; chain=True
    for phase in plan:
        marker=root/"phase_manifests"/f"phase_{phase.phase}.json"; errors=marker_validation_errors(marker,phase,commit=commit,plan_digest=digest)
        if not errors and chain: valid.append(phase.phase)
        else:
            chain=False; invalid[phase.phase]=errors
            missing.extend(str(p) for p in (*phase.inputs,*phase.outputs) if not p.exists())
            if any("hashes mismatch" in e for e in errors): changed.append(phase.phase)
    earliest=next((p.phase for p in plan if p.phase not in valid),None)
    reusable={str(p):p in valid for p in (5,6)}
    return {"valid_completed_phases":valid,"invalid_markers":invalid,"missing_artifacts":sorted(set(missing)),"changed_input_hashes":changed,"earliest_phase_that_must_run":earliest,"checkpoint_phases_reusable":reusable,"can_safely_resume":root.is_dir() and bool(valid) and not any(p<=(earliest or 99) for p in changed)}


def main(argv: Sequence[str]|None=None) -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--through","--stop-after",dest="through",type=int,choices=range(10),default=1); parser.add_argument("--resume",action="store_true"); parser.add_argument("--inspect-resume",action="store_true"); parser.add_argument("--restart-from",type=int); parser.add_argument("--execute",action="store_true"); parser.add_argument("--device",default="cuda"); parser.add_argument("--run-root",type=Path)
    args=parser.parse_args(argv)
    commit=_git("rev-parse","HEAD"); root=args.run_root or Path("results/formal")/f"hard_locker_rerun_{commit[:12]}"; bank_paths=_scenario_banks(); plan=build_plan(output_root=root,commit=commit,train_manifest=bank_paths["train"],device=args.device); validate_plan(plan,root); digest=plan_hash(plan)
    provenance={"commit":commit,"plan_hash":digest,"python":sys.version,"platform":platform.platform(),"selected_agents":["assignment"],"reward_model_lineage":"regenerate: assignment schema v4"}
    selected=[p for p in plan if p.phase<=args.through]
    if args.inspect_resume:
        if not root.is_dir(): raise SystemExit(f"resume run root does not exist: {root}")
        report=inspect_resume(root,selected,commit=commit,digest=digest); print(json.dumps(report,indent=2))
        if report["can_safely_resume"] and not report["invalid_markers"]: print("HARD_LOCKER_RESUME_STATE_VALID")
        return
    summary={"output_root":str(root),"provenance":provenance,"plan":[asdict(p) for p in selected],"training_jobs":6,"benchmark_rows":600,
             "requested_device":args.device,"require_cuda":args.device != "cpu","gpu_readiness_output":str(root/"gpu_readiness.json"),
             "reward_model_device":args.device,"mappo_device":args.device,
             "markers":["HARD_LOCKER_RERUN_PLAN_ISOLATED","ASSIGNMENT_PREFERENCE_PATH_VALID","ARTIFACT_AWARE_CONFIG_RESOLUTION_ENABLED","PAIRED_STATISTICS_PHASE_CONFIGURED","GPU_BACKED_FORMAL_TRAINING_CONFIGURED"]}; print(json.dumps(summary,indent=2,default=str))
    if not args.execute: return
    if _git("status","--porcelain"): raise SystemExit("hard-locker rerun requires a clean repository")
    if args.resume:
        if not root.is_dir(): raise SystemExit(f"resume requires existing run root: {root}")
        try: stored=json.loads((root/"provenance.json").read_text())
        except Exception as exc: raise SystemExit(f"resume requires valid provenance: {exc}")
        if stored != provenance: raise SystemExit("resume provenance (commit, plan, or selected-agent scope) does not match")
    elif root.exists(): raise SystemExit(f"refusing to overwrite {root}; use --resume")
    else: root.mkdir(parents=True); atomic_write_json(root/"provenance.json",provenance)
    if args.restart_from is not None:
        if not args.resume: raise SystemExit("--restart-from requires --resume")
        try: invalidated=invalidate_markers_from(root,args.restart_from)
        except ValueError as exc: raise SystemExit(str(exc))
        print(f"Invalidated phases: {invalidated}; artifacts were preserved")
    context=ExperimentContext(commit,root,device=args.device); restore_phase_0_context(context,bank_paths); completed=[]
    for phase in selected:
        marker=root/"phase_manifests"/f"phase_{phase.phase}.json"
        if args.resume and marker_is_valid(marker,phase,commit=commit,plan_digest=digest):
            restore_context_through_phase(root,phase.phase,context); completed.append(phase.phase); continue
        if args.resume and marker.exists() and phase.phase in (5,6,7,8): raise SystemExit(f"phase {phase.phase} marker is invalid; refusing to overwrite expensive artifacts: {marker_validation_errors(marker,phase,commit=commit,plan_digest=digest)}")
        _run_execute_phase(phase,context,digest); completed.append(phase.phase); update_experiment_artifact_manifest(context,completed)

if __name__=="__main__": main()
