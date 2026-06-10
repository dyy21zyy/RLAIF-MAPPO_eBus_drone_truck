"""JSONL persistence and schema validation for Stage 4 preference data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

CANDIDATE_FEATURE_KEYS = {
    "action_id", "action_name", "feasible_flag", "estimated_delivery_time", "estimated_lateness",
    "estimated_truck_distance", "estimated_truck_time", "estimated_bus_wait_time",
    "estimated_bus_linehaul_time", "estimated_drone_time", "estimated_locker_load_after_assignment",
    "estimated_station_power_margin", "infeasibility_reasons",
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    return count


def validate_assignment_state(record: dict[str, Any]) -> None:
    required = {"state_id", "episode_id", "current_time", "feature_schema_version", "assignment_features",
                "parcel", "system_state_summary", "station_states", "candidate_actions",
                "candidate_action_features", "action_mask"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"assignment state missing keys: {sorted(missing)}")
    if record["feature_schema_version"] != "v1" or not record["assignment_features"]:
        raise ValueError("invalid assignment feature schema")
    if len(record["candidate_actions"]) != len(record["action_mask"]):
        raise ValueError("candidate actions and action mask differ in length")
    for action in record["candidate_actions"]:
        features = record["candidate_action_features"].get(action["action_name"])
        if not isinstance(features, dict) or CANDIDATE_FEATURE_KEYS - features.keys():
            raise ValueError(f"invalid candidate features for {action['action_name']}")
        if features["feasible_flag"] != action["feasible"]:
            raise ValueError("candidate feasibility fields disagree")
        if not action["feasible"] and not features["infeasibility_reasons"]:
            raise ValueError("infeasible candidate lacks reasons")


def validate_prompt(record: dict[str, Any]) -> None:
    required = {"prompt_id", "state_id", "action_a", "action_b", "prompt_version", "prompt_text", "metadata"}
    if required - record.keys() or record["action_a"] == record["action_b"]:
        raise ValueError("invalid prompt record")
    text = record["prompt_text"]
    for phrase in ("Return only valid JSON", "chosen must be either action_a or action_b", '"confidence"'):
        if phrase not in text:
            raise ValueError(f"prompt missing requirement: {phrase}")


def validate_preference(record: dict[str, Any], confidence_threshold: float = 0.6) -> dict[str, Any]:
    action_a, action_b = record["action_a"], record["action_b"]
    if record.get("chosen") not in {action_a, action_b}:
        raise ValueError("chosen is not one of the compared actions")
    expected_rejected = action_b if record["chosen"] == action_a else action_a
    if record.get("rejected") != expected_rejected:
        raise ValueError("rejected is not the other compared action")
    confidence = float(record["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    record["confidence"] = confidence
    record["parser_status"] = "ok"
    record["validation_status"] = "valid"
    record["usable_for_training"] = confidence >= confidence_threshold
    return record
