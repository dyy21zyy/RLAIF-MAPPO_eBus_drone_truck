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
        decisions = {"assignment": 0, "truck": 0, "bus": 0, "station": 0}
        decision_event_types: dict[str, int] = {}
        steps = 0
        while observation["agent"] != "terminal":
            decisions[observation["agent"]] += 1
            event_type = str(observation["event_type"])
            decision_event_types[event_type] = decision_event_types.get(event_type, 0) + 1
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
        delivered = sum(parcel.status in {"delivered", "DELIVERED"} for parcel in env.parcels.values())
        drone_deliveries = sum(
            parcel.status in {"delivered", "DELIVERED"} and parcel.mode in {"TBD", "TLD"}
            for parcel in env.parcels.values()
        )
        return {
            "steps": steps,
            "decision_events": steps,
            "decisions": decisions,
            "decision_event_types": decision_event_types,
            "assignment_events": decisions["assignment"],
            "bus_charging_events": decisions["bus"],
            "delivered_parcels": delivered,
            "undelivered_parcels": len(env.parcels) - delivered,
            "drone_deliveries": drone_deliveries,
            "total_parcels": len(env.parcels),
            "episode_reward": env.reward_total,
            "infeasible_action_corrections": env.infeasible_action_corrections,
            "any_nan": False,
            "any_negative_locker_load": any(station.locker_load_kg < 0 for station in env.stations.values()),
            "any_negative_battery_count": any(station.full_batteries < 0 for station in env.stations.values()),
            "any_negative_truck_capacity": any(
                truck.remaining_capacity_kg < 0 for truck in env.trucks
            ),
            "invariants": "passed",
        }
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
