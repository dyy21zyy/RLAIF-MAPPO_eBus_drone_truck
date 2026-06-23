"""Collect assignment decision states from the Stage 3 event-driven environment."""

from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Any

from data_pipeline.build_instance import build_instance
from envs import DynamicDeliveryEnv, first_feasible_policy
from envs.state_builder import (action_name, build_assignment_features, build_candidate_action_features,
                                build_station_states, build_system_summary, assignment_feature_names)
from rlaif.preference_dataset import validate_assignment_state, write_jsonl


def build_assignment_state(env: DynamicDeliveryEnv, episode_id: int) -> dict[str, Any]:
    if env.current_decision is None or env.current_decision.agent != "assignment":
        raise ValueError("assignment state can only be built at an assignment decision")
    parcel_id = env.current_decision.event.payload["parcel_id"]
    parcel = env.parcels[parcel_id]
    row = next(item for item in env.parcel_rows if item["parcel_id"] == parcel_id)
    mask = list(env.current_decision.action_mask)
    candidates = []
    features = {}
    for action_id, feasible in enumerate(mask):
        name = action_name(env, action_id)
        candidates.append({"action_id": action_id, "action_name": name, "feasible": bool(feasible)})
        features[name] = build_candidate_action_features(env, parcel, action_id, bool(feasible))
    record = {
        "state_id": f"episode_{episode_id:04d}:{parcel_id}:{env.now_min:.6f}",
        "episode_id": episode_id,
        "current_time": float(env.now_min),
        "feature_schema_version": "v2",
        "assignment_features": build_assignment_features(env, parcel),
        "assignment_feature_names": list(assignment_feature_names(env.station_ids)),
        "parcel": {
            "parcel_id": parcel_id,
            "release_time": parcel.release_time_min,
            "deadline": parcel.deadline_min,
            "deadline_remaining": max(0.0, parcel.deadline_min - env.now_min),
            "weight": parcel.weight_kg,
            "volume": float(row["volume"]),
            "priority": row["priority"],
            "customer_lat": float(row["customer_lat"]),
            "customer_lon": float(row["customer_lon"]),
            "drone_feasible": parcel.drone_feasible,
        },
        "system_state_summary": build_system_summary(env, parcel),
        "data_sources": getattr(env, "data_sources", {}),
        "station_states": build_station_states(env, parcel),
        "candidate_actions": candidates,
        "candidate_action_features": features,
        "action_mask": mask,
    }
    validate_assignment_state(record)
    return record


def collect_assignment_states(config_path: str | Path, episodes: int, output: str | Path,
                              fallback: bool = True, seed: int = 42) -> list[dict[str, Any]]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    with tempfile.TemporaryDirectory(prefix="stage4-instance-") as directory:
        instance = build_instance(config_path, fallback=fallback, output_root=directory)
        env = DynamicDeliveryEnv(Path(instance["output_directory"]) / "instance.json")
        rng = random.Random(seed)
        records = []
        for episode_id in range(episodes):
            observation, _ = env.reset(seed=seed + episode_id)
            while observation["agent"] != "terminal":
                if observation["agent"] == "assignment":
                    records.append(build_assignment_state(env, episode_id))
                    feasible = [index for index, allowed in enumerate(observation["action_mask"]) if allowed]
                    action = rng.choice(feasible)
                else:
                    # Stage 3's configured deterministic baseline is no charging (index zero).
                    action = 0 if observation["action_mask"][0] else first_feasible_policy(observation)
                observation, _reward, terminated, truncated, _info = env.step(action)
                if terminated or truncated:
                    break
    write_jsonl(output, records)
    return records
