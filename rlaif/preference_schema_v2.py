"""Version 2 multi-agent RLAIF preference-state schema helpers."""
from __future__ import annotations
from typing import Any

PREFERENCE_SCHEMA_VERSION = "v2"
CANDIDATE_SCHEMA_VERSION = "v2"
RLAIF_AGENT_TYPES = {"assignment", "truck", "bus", "station"}
AGENT_EVENT_TYPES = {
    "assignment": {"PARCEL_RELEASE"},
    "truck": {"TRUCK_AVAILABLE"},
    "bus": {"BUS_TERMINAL_DEPARTURE", "BUS_STATION_ARRIVAL"},
    "station": {"STATION_OPERATION"},
}
REQUIRED_STATE_KEYS = {
    "schema_version", "scenario_id", "episode_id", "state_id", "decision_id", "agent_type",
    "event_type", "event_time", "state_feature_names", "state_features", "candidate_feature_names",
    "candidate_actions", "candidate_features", "action_masks", "data_provenance",
    "collection_policy_provenance", "checkpoint_provenance",
}


def canonical_event_type(event_type: str) -> str:
    return {"BUS_DEPARTURE": "BUS_TERMINAL_DEPARTURE", "BUS_ARRIVAL": "BUS_STATION_ARRIVAL"}.get(str(event_type), str(event_type))


def validate_agent_event(agent_type: str, event_type: str) -> None:
    event_type = canonical_event_type(event_type)
    if agent_type not in RLAIF_AGENT_TYPES:
        raise ValueError(f"unsupported RLAIF agent_type: {agent_type}")
    if event_type not in AGENT_EVENT_TYPES[agent_type]:
        raise ValueError(f"event {event_type} is not compatible with {agent_type}")


def preference_state_from_observation(observation: dict[str, Any], *, scenario_id: str, episode_id: str | int,
                                      decision_id: str | int, collection_policy: dict[str, Any] | None = None,
                                      data_provenance: dict[str, Any] | None = None,
                                      checkpoint_provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    agent = str(observation.get("agent_id", observation.get("agent", "")))
    event = canonical_event_type(str(observation.get("event_type", "")))
    validate_agent_event(agent, event)
    candidates = list(observation.get("candidate_actions", []))
    features = [[float(v) for v in row] for row in observation.get("candidate_features", [])]
    masks = [bool(v) for v in observation.get("action_mask", [])]
    if len(candidates) != len(features) or len(candidates) != len(masks):
        raise ValueError("candidate actions, features, and masks must have equal length")
    state_id = f"{scenario_id}:{episode_id}:{decision_id}:{agent}:{event}"
    return {
        "schema_version": PREFERENCE_SCHEMA_VERSION,
        "scenario_id": str(scenario_id), "episode_id": str(episode_id), "state_id": state_id,
        "decision_id": str(decision_id), "agent_type": agent, "event_type": event,
        "event_time": float(observation.get("time_min", observation.get("event_time", 0.0))),
        "state_feature_names": list(observation.get("feature_names", [])),
        "state_features": [float(v) for v in observation.get("features", [])],
        "candidate_feature_names": list(observation.get("candidate_feature_names", [])),
        "candidate_actions": candidates, "candidate_features": features, "action_masks": masks,
        "data_provenance": dict(data_provenance or observation.get("data_provenance", {})),
        "collection_policy_provenance": dict(collection_policy or {"source": "unknown"}),
        "checkpoint_provenance": dict(checkpoint_provenance or {}),
    }


def validate_preference_state(record: dict[str, Any]) -> None:
    missing = REQUIRED_STATE_KEYS - set(record)
    if missing:
        raise ValueError(f"preference state missing keys: {sorted(missing)}")
    if record["schema_version"] != PREFERENCE_SCHEMA_VERSION:
        raise ValueError("unsupported preference schema_version")
    validate_agent_event(str(record["agent_type"]), str(record["event_type"]))
    if len(record["state_feature_names"]) != len(record["state_features"]):
        raise ValueError("state feature names and values differ")
    if len(record["candidate_actions"]) != len(record["candidate_features"]) or len(record["candidate_actions"]) != len(record["action_masks"]):
        raise ValueError("candidate arrays differ in length")
    if any(len(row) != len(record["candidate_feature_names"]) for row in record["candidate_features"]):
        raise ValueError("candidate feature row has wrong dimension")
