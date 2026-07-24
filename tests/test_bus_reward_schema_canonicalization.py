import math
import pytest

from envs.state_builder import (
    BUS_CHARGING_FEATURE_NAMES,
    BUS_LOADING_FEATURE_NAMES,
    CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES,
    COMMON_CANDIDATE_FEATURE_NAMES,
)
from rlaif.feature_alignment import align_named_features
from rlaif.reward_model_dataset import build_reward_pair_dataset
from tests.preference_v3_fixtures import rec


def bus_rec(event, split="train", pid=None):
    names = BUS_LOADING_FEATURE_NAMES if event == "BUS_TERMINAL_DEPARTURE" else BUS_CHARGING_FEATURE_NAMES
    values = [float(i + 1) for i in range(len(names))]
    pid = pid or f"{event}-{split}"
    return rec(
        preference_id=pid,
        scenario_id=f"scenario-{split}-{event}",
        state_id=f"state-{pid}",
        agent_type="bus",
        event_type=event,
        state_feature_names=list(names),
        state_features=values,
        candidate_a_feature_names=list(COMMON_CANDIDATE_FEATURE_NAMES),
        candidate_a_features=[1.0] * len(COMMON_CANDIDATE_FEATURE_NAMES),
        candidate_b_feature_names=list(COMMON_CANDIDATE_FEATURE_NAMES),
        candidate_b_features=[0.0] * len(COMMON_CANDIDATE_FEATURE_NAMES),
        original_candidate_a_id=f"a-{pid}",
        original_candidate_b_id=f"b-{pid}",
        displayed_first_candidate_id=f"a-{pid}",
        displayed_second_candidate_id=f"b-{pid}",
        dataset_split=split,
    )


def test_bus_mixed_event_specific_schemas_succeed_after_canonicalization():
    ds = build_reward_pair_dataset([
        bus_rec("BUS_TERMINAL_DEPARTURE"),
        bus_rec("BUS_STATION_ARRIVAL"),
    ], agent_type="bus", formal_mode=True, require_bus_event_coverage=True)
    assert len(ds) == 2
    assert ds.state_feature_names == CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES
    assert ds.candidate_feature_names == COMMON_CANDIDATE_FEATURE_NAMES


def test_terminal_departure_values_align_to_canonical_positions_and_zero_fill():
    vals = [10.0, 20.0, 30.0, 40.0]
    aligned = align_named_features(BUS_LOADING_FEATURE_NAMES, vals, CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES)
    assert aligned[CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES.index("time_norm")] == 10.0
    assert aligned[CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES.index("ready_parcel_count_norm")] == 20.0
    assert aligned[CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES.index("freight_load_norm")] == 30.0
    assert aligned[CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES.index("capacity_remaining_norm")] == 40.0
    assert aligned[CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES.index("soc_norm")] == 0.0


def test_station_arrival_values_align_to_canonical_positions_and_zero_fill():
    vals = [float(i + 1) for i in range(len(BUS_CHARGING_FEATURE_NAMES))]
    aligned = align_named_features(BUS_CHARGING_FEATURE_NAMES, vals, CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES)
    assert aligned[CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES.index("time_norm")] == vals[0]
    assert aligned[CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES.index("freight_load_norm")] == vals[5]
    assert aligned[CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES.index("ready_parcel_count_norm")] == 0.0
    assert aligned[CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES.index("capacity_remaining_norm")] == 0.0


def test_shared_fields_not_duplicated_and_order_is_deterministic():
    assert CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES == (
        "time_norm", "ready_parcel_count_norm", "freight_load_norm", "capacity_remaining_norm",
        "soc_norm", "delay_norm", "locker_load_norm", "full_batteries_norm",
        "total_onboard_passengers_norm", "remaining_passenger_capacity_norm",
        "current_stop_waiting_passengers_norm", "downstream_waiting_summary_norm",
        "boarding_count_norm", "alighting_count_norm",
        "current_waiting_passenger_minutes_norm",
        "current_onboard_additional_delay_passenger_minutes_norm",
        "expected_boarding_time_norm", "expected_alighting_time_norm",
    )
    assert len(CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES) == len(set(CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES))


@pytest.mark.parametrize("names,values,error", [
    (["x", "x"], [1, 2], "duplicate"),
    (["x"], [1, 2], "differ"),
    (["x"], [math.inf], "non-finite"),
    (["unknown"], [1], "unknown"),
])
def test_named_alignment_fails_closed(names, values, error):
    with pytest.raises(ValueError, match=error):
        align_named_features(names, values, ["x"])


def test_both_bus_events_remain_present_in_each_split():
    rows = [bus_rec(ev, split, pid=f"{ev}-{split}") for split in ("train", "validation", "test") for ev in ("BUS_TERMINAL_DEPARTURE", "BUS_STATION_ARRIVAL")]
    ds = build_reward_pair_dataset(rows, agent_type="bus", formal_mode=True, require_bus_event_coverage=True)
    seen = {split: set() for split in ("train", "validation", "test")}
    for row in rows:
        seen[row["dataset_split"]].add(row["event_type"])
    assert all(events == {"BUS_TERMINAL_DEPARTURE", "BUS_STATION_ARRIVAL"} for events in seen.values())
    assert len(ds) == 6
