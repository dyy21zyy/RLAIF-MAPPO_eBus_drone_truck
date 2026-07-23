from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEMPLATE = Path("configs/diagnostic/preformal_gate.template.yaml")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _scenario_bank(root: Path, split: str, count: int) -> Path:
    bank = root / "scenario_banks" / split
    bank.mkdir(parents=True, exist_ok=True)
    scenarios = []
    for idx in range(count):
        scenario = {
            "scenario_id": f"diagnostic_{split}_{idx}",
            "run_classification": "diagnostic",
            "experiment_stage": "preformal_diagnostic",
            "publication_eligible": False,
            "master_seed": 1000 + idx,
            "events": [
                "PARCEL_RELEASE",
                "TRUCK_AVAILABLE",
                "BUS_TERMINAL_DEPARTURE",
                "BUS_STATION_ARRIVAL",
                "STATION_OPERATION",
            ],
        }
        path = bank / f"scenario_{idx:03d}.json"
        _write_json(path, scenario)
        scenarios.append({"path": str(path), "sha256": _sha(path), "scenario_id": scenario["scenario_id"]})
    manifest = {
        "split": split,
        "scenario_count": count,
        "run_classification": "diagnostic",
        "experiment_stage": "preformal_diagnostic",
        "publication_eligible": False,
        "scenarios": scenarios,
    }
    manifest_path = bank / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _checkpoint(root: Path, agent: str) -> Path:
    path = root / "reward_models" / f"{agent}_reward_checkpoint.pt"
    _write_json(path, {"agent": agent, "run_classification": "diagnostic", "publication_eligible": False, "weights": [0.1, 0.2, 0.3]})
    return path


def build(output_root: Path, force: bool) -> Path:
    if output_root.exists():
        if not force:
            raise SystemExit(f"{output_root} exists; pass --force")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train = _scenario_bank(output_root, "train", 3)
    validation = _scenario_bank(output_root, "validation", 2)
    test = _scenario_bank(output_root, "test", 3)
    reward_scale = output_root / "reward_reference_scales.json"
    _write_json(reward_scale, {"run_classification": "diagnostic", "publication_eligible": False, "scale": 1.0, "sample_count": 8})
    reward_models = {agent: _checkpoint(output_root, agent) for agent in ("assignment", "truck", "bus", "station")}

    config_paths: dict[str, Path] = {}
    for name in ("environment_mappo", "assignment_ppo", "assignment_rlaif_mappo", "full_rlaif_mappo", "benchmark", "ablation", "sensitivity"):
        path = output_root / f"{name}.resolved.yaml"
        payload = {
            "run_classification": "diagnostic",
            "publication_eligible": False,
            "training_seed": 1,
            "total_episodes": 6,
            "rollout_episodes": 3,
            "train_scenario_bank": str(train),
            "validation_scenario_bank": str(validation),
            "test_scenario_bank": str(test),
            "reward_scale_artifact": str(reward_scale),
            "reward_checkpoints": {k: str(v) for k, v in reward_models.items()},
        }
        if name == "sensitivity":
            payload.update({"factor": "parcel_demand_multiplier", "values": [0.9, 1.1], "protocol": "fixed_policy"})
        if name == "ablation":
            payload.update({"methods": ["mappo_env", "mappo_rlaif_assignment"]})
        _write_yaml(path, payload)
        config_paths[name] = path

    cfg = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8")) or {}
    cfg.update({
        "output_root": str(output_root / "run"),
        "workload": {"train_scenarios": 3, "validation_scenarios": 2, "test_scenarios": 3, "training_seeds": [1], "mappo_total_episodes": 6, "rollout_episodes": 3},
        "scenario_bank_hashes": {"train": _sha(train), "validation": _sha(validation), "test": _sha(test)},
        "reward_scale_hash": _sha(reward_scale),
        "reward_model_hashes": {k: _sha(v) for k, v in reward_models.items()},
        "benchmark_config": str(config_paths["benchmark"]),
        "ablation_config": str(config_paths["ablation"]),
        "sensitivity_config": str(config_paths["sensitivity"]),
        "training_transition_counts": {"environment_mappo_training": 30, "assignment_rlaif_mappo_training": 30, "full_rlaif_mappo_training": 30},
        "optimizer_update_counts": {"environment_mappo_training": 1, "assignment_ppo_training": 1, "assignment_rlaif_mappo_training": 1, "full_rlaif_mappo_training": 1},
        "benchmark_row_counts": {"success": 6, "failed": 0},
        "ablation_job_counts": {"jobs": 2},
        "sensitivity_job_counts": {"jobs": 2},
        "paired_comparison_counts": {"comparisons": 3},
        "event_coverage_status": {"PARCEL_RELEASE": 3, "TRUCK_AVAILABLE": 3, "BUS_TERMINAL_DEPARTURE": 3, "BUS_STATION_ARRIVAL": 3, "STATION_OPERATION": 3},
        "reward_reconciliation_status": "passed",
        "fallback_count": 0,
    })
    cfg.setdefault("artifact_inventory", {}).setdefault("artifacts", {}).update({
        "train_scenario_bank": {"path": str(train)},
        "validation_scenario_bank": {"path": str(validation)},
        "test_scenario_bank": {"path": str(test)},
        "reward_scale_artifact": {"path": str(reward_scale)},
    })
    resolved = output_root / "preformal_gate.resolved.yaml"
    _write_yaml(resolved, cfg)
    print(resolved)
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="results/diagnostic/preformal_gate")
    parser.add_argument("--output", help="Backward-compatible resolved config path")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output_root = Path(args.output_root if args.output is None else Path(args.output).parent)
    build(output_root, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
