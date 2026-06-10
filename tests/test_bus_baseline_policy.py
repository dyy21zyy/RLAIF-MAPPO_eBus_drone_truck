"""Tests for fixed Stage 6 bus charging baselines."""

import pytest

from training.bus_baseline_policy import BusBaselinePolicy


DURATIONS = [0, 15, 30, 45, 60, 120]


def test_no_charge_and_uniform_30_respect_masks() -> None:
    mask = [True, True, False, True, False, False]
    assert BusBaselinePolicy("no_charge").select_action(mask, DURATIONS) == 0
    assert BusBaselinePolicy("uniform_30").select_action(mask, DURATIONS) == 1


def test_battery_threshold_respects_soc_limit_and_mask() -> None:
    policy = BusBaselinePolicy("battery_threshold", battery_threshold_soc=0.4, max_charge_seconds=60)
    mask = [True, False, True, True, False, True]
    assert policy.select_action(mask, DURATIONS, bus_soc=0.2) == 3
    assert policy.select_action(mask, DURATIONS, bus_soc=0.8) == 0


def test_bus_policy_rejects_all_zero_mask() -> None:
    with pytest.raises(ValueError, match="no feasible"):
        BusBaselinePolicy().select_action([False] * len(DURATIONS), DURATIONS)
