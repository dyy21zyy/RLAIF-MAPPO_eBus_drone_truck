"""Asynchronous one-transition-per-event rollout buffer for Stage 7 MAPPO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
from training.event_schema import validate_agent_event, decision_event_id, normalize_decision_event_type

VALID_AGENTS = {"assignment", "truck", "bus", "station"}


@dataclass
class AsyncTransition:
    agent_id: str
    local_obs: list[float]
    global_state: list[float]
    action: int
    action_mask: list[bool]
    candidate_features: list[list[float]]
    candidate_feature_names: tuple[str, ...]
    log_prob: float
    value: float
    reward: float
    done: bool
    next_global_state: list[float]
    event_type: str
    event_time: float
    event_type_id: int | None = None
    environment_reward: float = 0.0
    learned_reward_raw: float = 0.0
    learned_reward_normalized: float = 0.0
    learned_reward_clipped: float = 0.0
    learned_reward_weighted: float = 0.0
    total_reward: float | None = None
    used_rlaif_fallback: bool = False
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
        if transition.event_type_id is None:
            canonical = normalize_decision_event_type(transition.event_type)
        else:
            canonical = validate_agent_event(transition.agent_id, transition.event_type, transition.event_type_id)
        transition.event_type = canonical
        transition.event_type_id = decision_event_id(canonical)
        if transition.total_reward is None:
            transition.total_reward = float(transition.reward)
        if not transition.local_obs or not transition.global_state or not transition.next_global_state:
            raise ValueError("Transition observations and global states must be non-empty")
        if not transition.action_mask or not 0 <= transition.action < len(transition.action_mask):
            raise ValueError("Transition action and mask are inconsistent")
        if not transition.action_mask[transition.action]:
            raise ValueError("Cannot append an infeasible action")
        if len(transition.candidate_features) != len(transition.action_mask):
            raise ValueError("Candidate features must align with the action mask")
        for row in transition.candidate_features:
            if len(row) != len(transition.candidate_feature_names):
                raise ValueError("Candidate feature rows must match candidate_feature_names")
        nums = [transition.log_prob, transition.value, transition.reward, transition.event_time, transition.environment_reward, transition.learned_reward_raw, transition.learned_reward_normalized, transition.learned_reward_clipped, transition.learned_reward_weighted, transition.total_reward]
        if any(not np.isfinite(float(v)) for v in nums): raise ValueError("Transition numerical fields must be finite")
        if abs(float(transition.total_reward) - (float(transition.environment_reward) + float(transition.learned_reward_weighted))) > 1e-5 and (transition.environment_reward or transition.learned_reward_weighted): raise ValueError("Transition total_reward must equal environment_reward + learned_reward_weighted")
        self.transitions.append(transition)

    def by_agent(self, agent_id: str) -> list[AsyncTransition]:
        if agent_id not in VALID_AGENTS:
            raise ValueError(f"Unknown agent_id: {agent_id}")
        return [item for item in self.transitions if item.agent_id == agent_id]

    def compute_returns_and_advantages(
        self,
        gamma: float = 0.997,
        gae_lambda: float = 0.95,
        reference_time_unit: float = 1.0,
        per_agent_normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE along the real event stream, resetting at episode boundaries."""
        count = len(self.transitions)
        advantages = np.zeros(count, dtype=np.float32)
        gae = 0.0
        time_unit = max(float(reference_time_unit), 1e-8)
        for index in range(count - 1, -1, -1):
            item = self.transitions[index]
            next_item = self.transitions[index + 1] if index + 1 < count else None
            same_episode = next_item is not None and next_item.episode_id == item.episode_id
            nonterminal = 0.0 if item.done else 1.0
            next_value = next_item.value if same_episode and not item.done else 0.0
            elapsed = float(item.info.get("delta_time_min", max(0.0, float(next_item.event_time) - float(item.event_time)) if same_episode else time_unit))
            discount = float(gamma) ** (elapsed / time_unit)
            delta = item.reward + discount * nonterminal * next_value - item.value
            if not same_episode:
                gae = 0.0
            gae = delta + discount * float(gae_lambda) * nonterminal * gae
            advantages[index] = gae
        self.returns = advantages + np.asarray([item.value for item in self.transitions], dtype=np.float32)
        if count and per_agent_normalize:
            normalized = advantages.copy()
            for agent in VALID_AGENTS:
                idx = np.asarray([i for i,t in enumerate(self.transitions) if t.agent_id == agent], dtype=int)
                if len(idx):
                    vals = advantages[idx]
                    normalized[idx] = (vals - float(vals.mean())) / (float(vals.std()) + 1e-8)
            self.advantages = normalized
        elif count:
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
