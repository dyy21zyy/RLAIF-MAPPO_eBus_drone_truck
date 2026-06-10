"""Shared helpers for dependency-light Stage 8 baseline policies."""
from __future__ import annotations

from typing import Sequence


def feasible_indices(action_mask: Sequence[bool]) -> list[int]:
    return [index for index, feasible in enumerate(action_mask) if bool(feasible)]


def fallback_action(action_mask: Sequence[bool], preferred: int = 0) -> int:
    feasible = feasible_indices(action_mask)
    if preferred < len(action_mask) and bool(action_mask[preferred]):
        return preferred
    return feasible[0] if feasible else 0


def duration_action(action_mask: Sequence[bool], durations_sec: Sequence[float], target_sec: float) -> int:
    feasible = feasible_indices(action_mask)
    if not feasible:
        return 0
    return min(feasible, key=lambda index: (abs(float(durations_sec[index]) - target_sec), index))
