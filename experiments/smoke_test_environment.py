"""Offline Stage 3 gate: build fallback data and complete one MDP episode."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Sequence

from data_pipeline.build_instance import build_instance
from envs import DynamicDeliveryEnv, first_feasible_policy


def run_smoke_test(config_path: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    if output_root is None:
        temporary = tempfile.TemporaryDirectory(prefix="stage3-smoke-")
        output_root = temporary.name
    else:
        temporary = None
    try:
        instance = build_instance(config_path, fallback=True, output_root=output_root)
        env = DynamicDeliveryEnv(Path(instance["output_directory"]) / "instance.json")
        observation, _ = env.reset()
        decisions = {"assignment": 0, "bus": 0}
        steps = 0
        while observation["agent"] != "terminal":
            decisions[observation["agent"]] += 1
            observation, _reward, terminated, truncated, _info = env.step(first_feasible_policy(observation))
            steps += 1
            if steps > 10000:
                raise AssertionError("Stage 3 smoke episode exceeded 10,000 decisions")
            if env.check_invariants():
                raise AssertionError(env.check_invariants())
            if terminated or truncated:
                break
        assert env.terminated and not env.truncated
        assert decisions["assignment"] == instance["counts"]["parcels"]
        assert env.check_invariants() == []
        return {"steps": steps, "decisions": decisions, "delivered_parcels": sum(p.status == "delivered" for p in env.parcels.values()),
                "total_parcels": len(env.parcels), "episode_reward": env.reward_total, "invariants": "passed"}
    finally:
        if temporary is not None:
            temporary.cleanup()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_smoke_test(args.config, args.output_root)
    print("Stage 3 event-driven environment smoke test passed.")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
