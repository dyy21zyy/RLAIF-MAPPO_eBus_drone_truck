"""Stage 7 asynchronous MAPPO event and reward semantics."""

from __future__ import annotations

from typing import Any, Sequence

from training.reward_model_wrapper import RewardModelWrapper

EVENT_SCHEMA_VERSION = 1
OBSERVATION_SCHEMA_VERSION = 3
CANDIDATE_SCHEMA_VERSION = 2

ASSIGNMENT_AGENT = "assignment"
TRUCK_AGENT = "truck"
BUS_AGENT = "bus"
STATION_AGENT = "station"
VALID_AGENT_EVENTS = {
    ASSIGNMENT_AGENT: {"PARCEL_RELEASE"},
    TRUCK_AGENT: {"TRUCK_AVAILABLE"},
    BUS_AGENT: {"BUS_TERMINAL_DEPARTURE", "BUS_STATION_ARRIVAL", "BUS_DEPARTURE", "BUS_ARRIVAL"},
    STATION_AGENT: {"STATION_OPERATION"},
}
CANONICAL_EVENT_MAP = {"BUS_DEPARTURE": "BUS_TERMINAL_DEPARTURE", "BUS_ARRIVAL": "BUS_STATION_ARRIVAL"}
RLAIF_AGENT_TYPES = {ASSIGNMENT_AGENT, TRUCK_AGENT, BUS_AGENT, STATION_AGENT}


def validate_decision(agent_id: str, event_type: str) -> None:
    if agent_id not in VALID_AGENT_EVENTS or event_type not in VALID_AGENT_EVENTS[agent_id]:
        raise ValueError(f"Invalid asynchronous decision pairing: {agent_id}/{event_type}")


def transition_reward(
    agent_id: str,
    env_reward: float,
    reward_wrapper: RewardModelWrapper,
    *,
    lambda_rlaif: float = 1.0,
    state_features: Sequence[float] | None = None,
    action_features: Sequence[float] | None = None,
    action_id: int | None = None,
    event_type: str | None = None,
) -> tuple[float, float]:
    """Return ``(total, learned_reward)`` for all four RLAIF-capable agents."""
    if agent_id not in RLAIF_AGENT_TYPES:
        raise ValueError(f"Unknown agent_id: {agent_id}")
    learned = 0.0
    if reward_wrapper.enabled:
        if state_features is None or action_features is None or action_id is None:
            raise ValueError("Enabled RLAIF requires assignment state/action features")
        try:
            learned = reward_wrapper.score(state_features, action_features, int(action_id), event_type=event_type)
        except TypeError:
            learned = reward_wrapper.score(state_features, action_features, int(action_id))
    return float(env_reward) + float(lambda_rlaif) * learned, learned


def reward_decomposition(info: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in info.get("reward_components", {}).items()}
