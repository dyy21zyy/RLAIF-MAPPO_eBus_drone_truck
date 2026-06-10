"""Tests for assignment-only PPO rollout storage."""

import numpy as np
import pytest

from training.ppo_buffer import AssignmentTransition, PPOBuffer


def transition(reward: float, done: bool = False) -> AssignmentTransition:
    return AssignmentTransition(
        obs=[0.0, 1.0], action=0, action_mask=[True, False], log_prob=-0.2,
        value=0.5, reward=reward, done=done, next_obs=[0.1, 0.9],
        info={"agent": "assignment"}, episode_id=1, event_time=2.0, parcel_id="p1",
        chosen_action_name="TD", r_env=reward, r_rlaif=0.0, r_total=reward,
    )


def test_buffer_add_returns_gae_and_clear() -> None:
    buffer = PPOBuffer()
    buffer.add(transition(1.0))
    buffer.add(transition(2.0, done=True))
    returns, advantages = buffer.compute_returns_and_advantages(0.99, 0.95)
    assert len(buffer) == 2
    assert returns.shape == advantages.shape == (2,)
    assert np.isfinite(returns).all()
    assert returns[1] == pytest.approx(2.0)
    buffer.clear()
    assert len(buffer) == 0
    assert buffer.returns.size == 0


def test_buffer_rejects_bus_transitions() -> None:
    item = transition(1.0)
    item.info["agent"] = "bus"
    with pytest.raises(ValueError, match="assignment transitions only"):
        PPOBuffer().add(item)
