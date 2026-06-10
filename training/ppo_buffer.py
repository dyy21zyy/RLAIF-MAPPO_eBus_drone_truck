"""Assignment-only rollout storage and generalized advantage estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np


@dataclass
class AssignmentTransition:
    obs: list[float]
    action: int
    action_mask: list[bool]
    log_prob: float
    value: float
    reward: float
    done: bool
    next_obs: list[float]
    info: dict[str, Any] = field(default_factory=dict)
    episode_id: int = 0
    event_time: float = 0.0
    parcel_id: str = ""
    chosen_action_name: str = ""
    r_env: float = 0.0
    r_rlaif: float = 0.0
    r_total: float = 0.0


class PPOBuffer:
    """Store only assignment decisions; bus transitions are rejected explicitly."""

    def __init__(self) -> None:
        self.transitions: list[AssignmentTransition] = []
        self.advantages = np.empty(0, dtype=np.float32)
        self.returns = np.empty(0, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.transitions)

    def add(self, transition: AssignmentTransition | None = None, **kwargs: Any) -> None:
        item = transition or AssignmentTransition(**kwargs)
        if item.info.get("agent", "assignment") != "assignment":
            raise ValueError("Stage 6 PPOBuffer accepts assignment transitions only")
        numeric = (item.log_prob, item.value, item.reward, item.r_env, item.r_rlaif, item.r_total)
        if not all(np.isfinite(float(value)) for value in numeric):
            raise ValueError("Transition contains a non-finite numeric value")
        if len(item.obs) == 0 or len(item.action_mask) == 0:
            raise ValueError("Transition observation and action mask must be non-empty")
        self.transitions.append(item)

    def clear(self) -> None:
        self.transitions.clear()
        self.advantages = np.empty(0, dtype=np.float32)
        self.returns = np.empty(0, dtype=np.float32)

    def compute_returns_and_advantages(
        self, gamma: float, gae_lambda: float, last_value: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.transitions:
            raise ValueError("Cannot compute GAE for an empty buffer")
        advantages = np.zeros(len(self), dtype=np.float32)
        gae = 0.0
        next_value = float(last_value)
        for index in range(len(self) - 1, -1, -1):
            item = self.transitions[index]
            nonterminal = 0.0 if item.done else 1.0
            delta = item.reward + gamma * next_value * nonterminal - item.value
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[index] = gae
            next_value = item.value
        self.advantages = advantages
        self.returns = advantages + np.asarray([item.value for item in self.transitions], dtype=np.float32)
        return self.returns, self.advantages

    def minibatch_indices(self, batch_size: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = rng.permutation(len(self))
        for start in range(0, len(indices), batch_size):
            yield indices[start : start + batch_size]
