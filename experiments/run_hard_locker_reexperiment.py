"""Fail-closed, resumable command plan for the hard-locker formal rerun.

The GPU phases are intentionally explicit subprocesses: no legacy checkpoint or
result directory is selected implicitly.  Use ``--through`` to stop at a phase.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


PHASES = {
    1: [[sys.executable, "-m", "pytest", "-q", "tests/test_hard_locker_capacity.py"],
        [sys.executable, "-m", "experiments.smoke_test_environment", "--config", "configs/shanghai_small.yaml"]],
    2: [[sys.executable, "-m", "experiments.generate_formal_multiagent_preferences", "--config", "configs/paper/rlaif_preference_generation.yaml", "--output-root", "data/preference/hard_locker"],
        [sys.executable, "-m", "experiments.train_multi_agent_reward_models", "--config", "configs/paper/train_reward_assignment.yaml", "--agent", "assignment"]],
    3: [[sys.executable, "-m", "experiments.estimate_reward_reference_scales", "--scenario-bank", "results/formal/scenario_banks/train/manifest.json", "--config", "configs/paper/reward_scale_estimation.yaml", "--output", "results/formal/reward_scales/final_reward_reference_scales.json"]],
    4: [[sys.executable, "-m", "experiments.freeze_final_experiment_parameters", "--config", "configs/paper/final_experiment_freeze.template.yaml", "--output", "results/formal/final_experiment_freeze.yaml"]],
    5: [[sys.executable, "-m", "experiments.train_mappo_async", "--config", "configs/paper/train_mappo_env.yaml", "--seed", str(seed)] for seed in (1, 2, 3)],
    6: [[sys.executable, "-m", "experiments.train_mappo_async", "--config", "configs/paper/train_mappo_rlaif_assignment.yaml", "--seed", str(seed)] for seed in (1, 2, 3)],
    7: [[sys.executable, "-m", "experiments.validate_formal_experiment_readiness", "--config", "configs/paper/benchmark.yaml", "--strict"]],
    8: [[sys.executable, "-m", "experiments.run_paper_benchmark", "--config", "configs/paper/benchmark.yaml"]],
    9: [[sys.executable, "-m", "experiments.aggregate_paper_results", "--input", "results/formal/benchmark", "--output", "results/formal/paired_analysis.json"]],
}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", type=int, choices=range(0, 10), default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true", help="execute rather than print the plan")
    args = parser.parse_args()
    if _git("status", "--porcelain"):
        raise SystemExit("hard-locker rerun requires a clean repository")
    commit = _git("rev-parse", "HEAD")
    root = Path("results/formal") / f"hard_locker_rerun_{commit[:12]}"
    if root.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite {root}; use --resume")
    provenance = {"commit": commit, "python": sys.version, "platform": platform.platform(),
                  "reward_model_lineage": "regenerate: assignment features changed under observation/candidate schema v4"}
    plan = [{"phase": phase, "commands": commands} for phase, commands in PHASES.items() if phase <= args.through]
    print(json.dumps({"output_root": str(root), "provenance": provenance, "plan": plan}, indent=2))
    if not args.execute:
        return
    root.mkdir(parents=True, exist_ok=args.resume)
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for item in plan:
        marker = root / f"phase_{item['phase']}.complete"
        if args.resume and marker.exists():
            continue
        for command in item["commands"]:
            subprocess.run(command, check=True)
        marker.write_text(hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest() + "\n")


if __name__ == "__main__":
    main()
