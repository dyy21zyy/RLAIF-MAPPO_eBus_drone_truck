"""Stage 7 asynchronous MAPPO event and reward semantics."""

from __future__ import annotations

from typing import Any, Sequence

from training.reward_model_wrapper import RewardModelWrapper

ASSIGNMENT_AGENT = "assignment"
BUS_AGENT = "bus"
ASSIGNMENT_EVENT = "PARCEL_ARRIVAL"
BUS_EVENT = "BUS_ARRIVAL"


def validate_decision(agent_id: str, event_type: str) -> None:
    expected = {ASSIGNMENT_AGENT: ASSIGNMENT_EVENT, BUS_AGENT: BUS_EVENT}
    if agent_id not in expected or event_type != expected[agent_id]:
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
) -> tuple[float, float]:
    """Return ``(total, learned_reward)``; learned reward is assignment-only."""
    if agent_id == BUS_AGENT:
        return float(env_reward), 0.0
    if agent_id != ASSIGNMENT_AGENT:
        raise ValueError(f"Unknown agent_id: {agent_id}")
    learned = 0.0
    if reward_wrapper.enabled:
        if state_features is None or action_features is None or action_id is None:
            raise ValueError("Enabled RLAIF requires assignment state/action features")
        learned = reward_wrapper.score(state_features, action_features, int(action_id))
    return float(env_reward) + float(lambda_rlaif) * learned, learned


def reward_decomposition(info: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in info.get("reward_components", {}).items()}
