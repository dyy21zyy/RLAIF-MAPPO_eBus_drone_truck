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
from typing import Any, Callable, Sequence

import yaml

TOKENS = ("REPLACE_WITH_", "PLACEHOLDER", "TODO_HASH")
SCOPE = ("assignment",)
SCHEMA_VERSION = 4


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
    return {"path": str(path), "artifact_hash": canonical, "hash": canonical, "file_sha256": sha256_file(path), "training_bank_hash": train_bank_hash,
            "validation_status": data["validation_status"]}


def materialize_initial_configs(root: Path, device: str = "cuda") -> Path:
    """Materialize only the reward-model config, whose inputs are already known."""
    destination = root / "resolved_configs" / "train_reward_assignment.yaml"
    config = _load_data(Path("configs/paper/train_reward_assignment.yaml"))
    config.setdefault("training", {}).update({"device": device, "require_cuda": True})
    _dump_yaml(destination, config)
    return destination

def gpu_readiness(path: Path, requested: str) -> dict[str, Any]:
    import torch
    from training.device import resolve_torch_device
    device=resolve_torch_device(requested, require_cuda=True)
    index=device.index; props=torch.cuda.get_device_properties(device)
    a=torch.ones((8,8),device=device); smoke=float((a@a).sum().item())
    record={"marker":"FORMAL_CUDA_READY","requested_device":requested,"resolved_device":str(device),"torch_version":torch.__version__,"cuda_runtime_version":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"cuda_device_count":torch.cuda.device_count(),"selected_cuda_device_index":index,"cuda_device_name":props.name,"compute_capability":list(torch.cuda.get_device_capability(device)),"total_device_memory":props.total_memory,"tensor_smoke_result":smoke}
    path.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n"); return record


def resolve_training_configs(root: Path, banks: dict[str, dict[str, Any]], reward_model: dict[str, Any], scale: dict[str, Any], device: str = "cuda") -> tuple[Path, Path]:
    resolved = root / "resolved_configs"
    outputs = []
    for method, template in (("mappo_env", "train_mappo_env.yaml"), ("mappo_rlaif_assignment", "train_mappo_rlaif_assignment.yaml")):
        cfg = _load_data(Path("configs/paper") / template)
        cfg.setdefault("training", {}).update({"device":device,"require_cuda":True})
        cfg["scenario_bank"] = {"manifest": banks["train"]["path"], "bank_hash": banks["train"]["bank_hash"]}
        cfg.setdefault("env", {}).update({"scenario_bank_manifest": banks["train"]["path"], "expected_split":"train", "expected_bank_hash":banks["train"]["bank_hash"], "scenario_sampling_mode":"shuffled_cycle"})
        cfg["reward"]["scale_artifact"] = scale["path"]
        cfg["reward"]["scale_artifact_hash"] = scale["hash"]
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
    found: dict[str, dict[int, dict[str, Any]]] = {}
    for method in ("mappo_env", "mappo_rlaif_assignment"):
        found[method] = {}
        for seed in (1, 2, 3):
            run = root / method / f"seed_{seed}"
            manifest = run / "training_run_manifest.json"
            if not manifest.is_file(): raise ValueError(f"formal training manifest missing: {manifest}")
            data = json.loads(manifest.read_text()); candidate=Path(data.get("checkpoint_path", ""))
            if data.get("status") != "complete" or not candidate.is_file() or data.get("checkpoint_file_sha256") != sha256_file(candidate): raise ValueError(f"invalid training manifest: {manifest}")
            found[method][seed] = validate_policy_checkpoint(candidate, method=method, seed=seed,
                commit=commit, banks=banks, reward_model=reward_model, scale=scale)
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
    pref_root = root / "preferences"; pref_file = assignment_preference_path(pref_root)
    reward = root / "reward_models" / "reward_assignment.pt"
    scale = root / "reward_scales" / "final_reward_reference_scales.json"
    resolved = root / "resolved_configs"; benchmark = root / "benchmark"
    env_roots = tuple(root / "mappo_env" / f"seed_{seed}" for seed in (1,2,3))
    rlaif_roots = tuple(root / "mappo_rlaif_assignment" / f"seed_{seed}" for seed in (1,2,3))
    return [
      PhaseSpec(0,"initial_artifact_resolution",tuple(),(resolved/"train_reward_assignment.yaml",root/"gpu_readiness.json")),
      PhaseSpec(1,"verification",((py,"-m","pytest","-q","tests/test_hard_locker_capacity.py"),), (root/"verification.json",)),
      PhaseSpec(2,"assignment_preferences_and_reward_model",(
        (py,"-m","experiments.generate_formal_multiagent_preferences","--config","configs/paper/rlaif_preference_generation.yaml","--output-root",str(pref_root),"--agents","assignment","--cache-dir",str(pref_root/"evaluator_cache")),
        (py,"-m","experiments.train_multi_agent_reward_models","--preferences",str(pref_file),"--config",str(resolved/"train_reward_assignment.yaml"),"--agent","assignment","--output",str(reward),"--device",device)),
        (pref_file,pref_root/"preference_manifest.json",reward), validation="assignment_preference_and_reward"),
      PhaseSpec(3,"reward_scales_and_training_config_resolution",((py,"-m","experiments.estimate_reward_reference_scales","--scenario-bank",str(train_manifest or _scenario_banks()["train"]),"--config","configs/paper/reward_scale_estimation.yaml","--output",str(scale)),), (scale,resolved/"train_mappo_env.yaml",resolved/"train_mappo_rlaif_assignment.yaml"), inputs=(reward,pref_file)),
      PhaseSpec(4,"placeholder_free_pretraining_gate",tuple(),(resolved/"train_mappo_env.yaml",resolved/"train_mappo_rlaif_assignment.yaml")),
      PhaseSpec(5,"mappo_env_training",tuple((py,"-m","experiments.train_mappo_async","--config",str(resolved/"train_mappo_env.yaml"),"--seed",str(seed),"--output-root",str(env_roots[seed-1])) for seed in (1,2,3)),env_roots),
      PhaseSpec(6,"rlaif_mappo_training",tuple((py,"-m","experiments.train_mappo_async","--config",str(resolved/"train_mappo_rlaif_assignment.yaml"),"--seed",str(seed),"--output-root",str(rlaif_roots[seed-1])) for seed in (1,2,3)),rlaif_roots),
      PhaseSpec(7,"checkpoint_resolution_and_final_freeze",((py,"-m","experiments.validate_formal_experiment_readiness","--config",str(resolved/"benchmark.yaml"),"--strict"),(py,"-m","experiments.freeze_final_experiment_parameters","--config",str(resolved/"final_experiment_freeze.yaml"),"--output",str(root/"final_experiment_freeze.json"))), (resolved/"benchmark.yaml",resolved/"final_experiment_freeze.yaml",root/"final_experiment_freeze.json")),
      PhaseSpec(8,"benchmark_600_rows",((py,"-m","experiments.run_paper_benchmark","--config",str(resolved/"benchmark.yaml"),"--output-root",str(benchmark)),), (benchmark/"episode_results.jsonl",)),
      PhaseSpec(9,"paired_analysis",((py,"-m","experiments.analyze_hard_locker_paired_results","--input",str(benchmark/"episode_results.jsonl"),"--output-dir",str(root/"paired_analysis")),), (root/"paired_analysis"/"analysis_manifest.json",)),
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


def phase_artifact_state(phase: PhaseSpec, *, commit: str, plan_digest: str, selected_agents: tuple[str,...] = SCOPE) -> dict[str, Any]:
    def hashes(paths: tuple[Path,...]) -> dict[str,str]:
        result = {}
        for path in paths:
            if path.is_file(): result[str(path)] = sha256_file(path)
            elif path.is_dir():
                result[str(path)] = hashlib.sha256(json.dumps({str(p.relative_to(path)):sha256_file(p) for p in sorted(path.rglob('*')) if p.is_file()},sort_keys=True).encode()).hexdigest()
            else: result[str(path)] = "MISSING"
        return result
    return {"phase":phase.phase,"name":phase.name,"commit":commit,"plan_hash":plan_digest,
            "selected_agents":list(selected_agents),"input_hashes":hashes(phase.inputs),"output_hashes":hashes(phase.outputs)}


def marker_is_valid(marker: Path, phase: PhaseSpec, *, commit: str, plan_digest: str) -> bool:
    if not marker.is_file(): return False
    stored = json.loads(marker.read_text())
    return stored == phase_artifact_state(phase,commit=commit,plan_digest=plan_digest) and "MISSING" not in stored["output_hashes"].values()


def write_phase_marker(marker: Path, phase: PhaseSpec, *, commit: str, plan_digest: str,
                       validator: Callable[[], None] | None = None) -> None:
    if validator: validator()
    state = phase_artifact_state(phase,commit=commit,plan_digest=plan_digest)
    missing = [path for path, digest in state["output_hashes"].items() if digest == "MISSING"]
    if missing: raise ValueError(f"phase outputs missing: {missing}")
    marker.parent.mkdir(parents=True,exist_ok=True); marker.write_text(json.dumps(state,indent=2,sort_keys=True)+"\n")


def _git(*args: str) -> str: return subprocess.check_output(["git",*args],text=True).strip()


def _scenario_banks() -> dict[str, Path]:
    candidates = {
      "train": (Path("results/formal/scenario_banks/train/manifest.json"),Path("results/formal/scenarios/train/scenario_bank_manifest.json")),
      "validation": (Path("results/formal/scenario_banks/validation/manifest.json"),Path("results/formal/scenarios/validation/scenario_bank_manifest.json")),
      "test": (Path("results/formal/scenario_banks/test/manifest.json"),Path("data/scenarios/test/scenario_bank_manifest.json")),
    }
    return {split: next((p for p in paths if p.is_file()), paths[0]) for split,paths in candidates.items()}


def _run_execute_phase(phase: PhaseSpec, root: Path, commit: str, digest: str, context: dict[str,Any]) -> None:
    if phase.phase == 0:
        materialize_initial_configs(root, context["device"])
        gpu_readiness(root/"gpu_readiness.json", context["device"])
    elif phase.phase == 1:
        subprocess.run(phase.commands[0],check=True)
        (root/"verification.json").write_text(json.dumps({"status":"pass"})+"\n")
    elif phase.phase == 2:
        subprocess.run(phase.commands[0],check=True)
        context["preference"] = validate_assignment_preferences(root/"preferences")
        subprocess.run(phase.commands[1],check=True)
        context["reward_model"] = validate_reward_checkpoint(root/"reward_models"/"reward_assignment.pt",context["preference"])
    elif phase.phase == 3:
        subprocess.run(phase.commands[0],check=True)
        context["scale"] = validate_reward_scale(root/"reward_scales"/"final_reward_reference_scales.json",context["banks"]["train"]["bank_hash"])
        resolve_training_configs(root,context["banks"],context["reward_model"],context["scale"],context["device"])
    elif phase.phase == 4:
        for path in phase.outputs: assert_no_placeholders(_load_data(path),label=str(path))
    elif phase.phase in (5,6,8,9):
        for command in phase.commands: subprocess.run(command,check=True)
    elif phase.phase == 7:
        context["checkpoints"] = discover_policy_checkpoints(root,commit,context["banks"],context["reward_model"],context["scale"])
        resolve_benchmark_config(root,context["banks"],context["reward_model"],context["checkpoints"])
        resolve_final_freeze_config(root,context["banks"],context["reward_model"],context["scale"],context["checkpoints"])
        for command in phase.commands: subprocess.run(command,check=True)
    write_phase_marker(root/"phase_manifests"/f"phase_{phase.phase}.json",phase,commit=commit,plan_digest=digest)

def restore_context_through_phase(root: Path, phase: int, context: dict[str,Any], commit: str) -> None:
    """Deterministically reload validated artifacts when resume skips phases."""
    if phase >= 2:
        context["preference"]=validate_assignment_preferences(root/"preferences")
        context["reward_model"]=validate_reward_checkpoint(root/"reward_models"/"reward_assignment.pt",context["preference"])
    if phase >= 3:
        context["scale"]=validate_reward_scale(root/"reward_scales"/"final_reward_reference_scales.json",context["banks"]["train"]["bank_hash"])
        for name in ("train_mappo_env.yaml","train_mappo_rlaif_assignment.yaml"):
            assert_no_placeholders(_load_data(root/"resolved_configs"/name),label=name)
    if phase >= 6:
        context["checkpoints"]=discover_policy_checkpoints(root,commit,context["banks"],context["reward_model"],context["scale"])


def main(argv: Sequence[str] | None = None) -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--through",type=int,choices=range(0,10),default=1); parser.add_argument("--resume",action="store_true"); parser.add_argument("--execute",action="store_true"); parser.add_argument("--device",default="cuda")
    args=parser.parse_args(argv)
    if args.execute and _git("status","--porcelain"): raise SystemExit("hard-locker rerun requires a clean repository")
    commit=_git("rev-parse","HEAD"); root=Path("results/formal")/f"hard_locker_rerun_{commit[:12]}"
    bank_paths=_scenario_banks(); plan=build_plan(output_root=root,commit=commit,train_manifest=bank_paths["train"],device=args.device); validate_plan(plan,root); digest=plan_hash(plan)
    provenance={"commit":commit,"plan_hash":digest,"python":sys.version,"platform":platform.platform(),"selected_agents":["assignment"],"reward_model_lineage":"regenerate: assignment schema v4"}
    selected=[phase for phase in plan if phase.phase<=args.through]
    pref=assignment_preference_path(root/"preferences")
    summary={"output_root":str(root),"provenance":provenance,"plan":[asdict(p) for p in selected],"training_jobs":6,"benchmark_rows":600,
      "requested_device":args.device,"resolved_device_policy":args.device,"gpu_readiness_output":str(root/"gpu_readiness.json"),"assignment_preference_jsonl":str(pref),"reward_model_checkpoint":str(root/"reward_models"/"reward_assignment.pt"),"reward_scale_artifact":str(root/"reward_scales"/"final_reward_reference_scales.json"),
      "policy_checkpoint_roots":[str(root/m/f"seed_{s}") for m in ("mappo_env","mappo_rlaif_assignment") for s in (1,2,3)],"benchmark_output":str(root/"benchmark"/"episode_results.jsonl"),"paired_analysis_output":str(root/"paired_analysis"),
      "markers":["HARD_LOCKER_RERUN_PLAN_ISOLATED","ASSIGNMENT_PREFERENCE_PATH_VALID","ARTIFACT_AWARE_CONFIG_RESOLUTION_ENABLED","PAIRED_STATISTICS_PHASE_CONFIGURED","GPU_BACKED_FORMAL_TRAINING_CONFIGURED"]}
    print(json.dumps(summary,indent=2,default=str))
    if not args.execute: return
    banks={split:validate_scenario_manifest(path,split) for split,path in bank_paths.items()}
    if root.exists():
        if not args.resume: raise SystemExit(f"refusing to overwrite {root}; use --resume")
        stored=json.loads((root/"provenance.json").read_text())
        if stored != provenance: raise SystemExit("resume provenance (commit, plan, or selected-agent scope) does not match")
    else:
        root.mkdir(parents=True); (root/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    context: dict[str,Any]={"banks":banks,"device":args.device}
    for phase in selected:
        marker=root/"phase_manifests"/f"phase_{phase.phase}.json"
        if args.resume and marker_is_valid(marker,phase,commit=commit,plan_digest=digest):
            restore_context_through_phase(root,phase.phase,context,commit)
            continue
        _run_execute_phase(phase,root,commit,digest,context)

if __name__=="__main__": main()
