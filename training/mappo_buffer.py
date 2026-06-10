"""Asynchronous one-transition-per-event rollout buffer for Stage 7 MAPPO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

VALID_AGENTS = {"assignment", "bus"}


@dataclass
class AsyncTransition:
    agent_id: str
    local_obs: list[float]
    global_state: list[float]
    action: int
    action_mask: list[bool]
    log_prob: float
    value: float
    reward: float
    done: bool
    next_global_state: list[float]
    event_type: str
    event_time: float
    info: dict[str, Any] = field(default_factory=dict)
    episode_id: int = 0


class AsyncMAPPOBuffer:
    """Stores the actual asynchronous event stream without inactive-agent rows."""

    def __init__(self) -> None:
        self.transitions: list[AsyncTransition] = []
        self.returns = np.empty(0, dtype=np.float32)
        self.advantages = np.empty(0, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.transitions)

    def append(self, transition: AsyncTransition) -> None:
        if transition.agent_id not in VALID_AGENTS:
            raise ValueError(f"agent_id must be one of {sorted(VALID_AGENTS)}")
        if not transition.local_obs or not transition.global_state or not transition.next_global_state:
            raise ValueError("Transition observations and global states must be non-empty")
        if not transition.action_mask or not 0 <= transition.action < len(transition.action_mask):
            raise ValueError("Transition action and mask are inconsistent")
        if not transition.action_mask[transition.action]:
            raise ValueError("Cannot append an infeasible action")
        self.transitions.append(transition)

    def by_agent(self, agent_id: str) -> list[AsyncTransition]:
        if agent_id not in VALID_AGENTS:
            raise ValueError(f"Unknown agent_id: {agent_id}")
        return [item for item in self.transitions if item.agent_id == agent_id]

    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE along the real event stream, resetting at episode boundaries."""
        count = len(self.transitions)
        advantages = np.zeros(count, dtype=np.float32)
        gae = 0.0
        for index in range(count - 1, -1, -1):
            item = self.transitions[index]
            next_item = self.transitions[index + 1] if index + 1 < count else None
            same_episode = next_item is not None and next_item.episode_id == item.episode_id
            nonterminal = 0.0 if item.done else 1.0
            next_value = next_item.value if same_episode and not item.done else 0.0
            delta = item.reward + float(gamma) * nonterminal * next_value - item.value
            if not same_episode:
                gae = 0.0
            gae = delta + float(gamma) * float(gae_lambda) * nonterminal * gae
            advantages[index] = gae
        self.returns = advantages + np.asarray([item.value for item in self.transitions], dtype=np.float32)
        if count:
            mean, std = float(advantages.mean()), float(advantages.std())
            self.advantages = (advantages - mean) / (std + 1e-8)
        else:
            self.advantages = advantages
        return self.returns, self.advantages

    def minibatch_indices(self, batch_size: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = rng.permutation(len(self))
        for start in range(0, len(indices), batch_size):
            yield indices[start:start + batch_size]

    def clear(self) -> None:
        self.transitions.clear()
        self.returns = np.empty(0, dtype=np.float32)
        self.advantages = np.empty(0, dtype=np.float32)
