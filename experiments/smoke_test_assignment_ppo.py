"""Dependency-light Stage 6 assignment PPO smoke test (no reward checkpoint required)."""

from __future__ import annotations

import math
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from utils.config import PROJECT_ROOT, load_config
from rlaif.torch_runtime import is_torch_runtime_available


def run_smoke_test() -> dict[str, Any]:
    if not is_torch_runtime_available():
        return {"skipped": True, "reason": "PyTorch is unavailable"}

    from training.ppo_trainer import load_assignment_checkpoint, train_assignment_ppo

    config = deepcopy(load_config(PROJECT_ROOT / "configs/train_assignment_ppo.yaml"))
    with tempfile.TemporaryDirectory(prefix="stage6_ppo_") as directory:
        root = Path(directory)
        config["training"].update({
            "total_episodes": 1, "rollout_episodes": 1, "ppo_epochs": 1, "batch_size": 128,
        })
        config["rlaif"].update({"enabled": False, "reward_model_checkpoint": str(root / "missing.pt")})
        config["output"] = {
            "checkpoint_path": str(root / "assignment_ppo.pt"),
            "training_log_path": str(root / "training.csv"),
            "eval_path": str(root / "evaluation.json"),
        }
        result = train_assignment_ppo(config, output_root=root / "instance")
        rows = result["rows"]
        if not rows or rows[0]["assignment_decision_count"] < 1:
            raise AssertionError("Smoke test collected no assignment transitions")
        if rows[0]["infeasible_action_count"] != 0:
            raise AssertionError("Masked assignment PPO selected an infeasible action")
        loss_fields = ("entropy", "policy_loss", "value_loss", "total_loss", "approx_kl", "clip_fraction")
        if not all(math.isfinite(float(rows[0][field])) for field in loss_fields):
            raise AssertionError("Smoke test produced a non-finite PPO loss")
        loaded, checkpoint = load_assignment_checkpoint(config["output"]["checkpoint_path"])
        if loaded.action_dim != result["model"].action_dim or checkpoint["stage"] != 6:
            raise AssertionError("Checkpoint save/load verification failed")
        return {
            "skipped": False,
            "assignment_transitions": int(rows[0]["assignment_decision_count"]),
            "losses_finite": True,
            "masks_respected": True,
            "checkpoint_round_trip": True,
            "rlaif_enabled": False,
        }


def main() -> int:
    result = run_smoke_test()
    if result["skipped"]:
        print(f"SKIP: {result['reason']}")
    else:
        print("Stage 6 assignment PPO smoke test passed.")
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
