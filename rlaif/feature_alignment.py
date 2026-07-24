"""Named feature-schema alignment utilities for reward-model inputs."""
from __future__ import annotations

import math
from typing import Sequence


def align_named_features(
    source_feature_names: Sequence[str],
    source_feature_values: Sequence[float],
    target_feature_names: Sequence[str],
    *,
    fill_value: float = 0.0,
    allow_unknown: bool = False,
) -> tuple[float, ...]:
    """Align source values into a canonical target schema by feature name."""
    source_names = tuple(str(x) for x in source_feature_names)
    target_names = tuple(str(x) for x in target_feature_names)
    if len(source_names) != len(source_feature_values):
        raise ValueError("source feature names and values differ")
    if len(set(source_names)) != len(source_names):
        raise ValueError("duplicate source feature names")
    if len(set(target_names)) != len(target_names):
        raise ValueError("duplicate target feature names")
    fill = float(fill_value)
    if not math.isfinite(fill):
        raise ValueError("fill value must be finite")
    values = tuple(float(v) for v in source_feature_values)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("source features contain non-finite values")
    unknown = tuple(name for name in source_names if name not in target_names)
    if unknown and not allow_unknown:
        raise ValueError(f"unknown source feature names: {list(unknown)}")
    mapping = dict(zip(source_names, values))
    return tuple(mapping.get(name, fill) for name in target_names)
