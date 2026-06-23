"""Smoke-test Stage 3 on an original-scale real-transit fixture instance."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Sequence

from data_pipeline.build_instance import build_instance
from envs import DynamicDeliveryEnv, first_feasible_policy
from experiments.original_scale_smoke_common import smoke_config


def run_smoke_test(config_path: str | Path, max_steps: int = 200) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="orig-scale-env-") as directory:
        root = Path(directory)
        cfg = smoke_config(config_path, root)
        instance = build_instance(cfg, fallback=False, output_root=root / "processed")
        env = DynamicDeliveryEnv(Path(instance["output_directory"]) / "instance.json")
        observation, _info = env.reset(seed=42)
        steps = 0
        while observation["agent"] != "terminal" and steps < max_steps:
            action = first_feasible_policy(observation)
            observation, _reward, terminated, truncated, _info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        metrics = env.get_metrics()
        if metrics["fallback_feasibility_events"] < 0:
            raise AssertionError("fallback_feasibility_events metric must be present")
        return {
            "steps": steps,
            "terminal": observation["agent"] == "terminal",
            "delivered_parcels": metrics["delivered_parcels"],
            "bus_charging_events": metrics["bus_charging_events"],
            "data_mode": instance["data_mode"],
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_smoke_test(args.config)
    print("Original-scale real-transit environment smoke test passed.")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
