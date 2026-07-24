import math
import pytest

from envs import first_feasible_policy
from envs.state_builder import (
    BUS_CHARGING_FEATURE_NAMES,
    BUS_EVENT_STATE_FEATURE_NAMES,
    BUS_LOADING_FEATURE_NAMES,
    CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES,
    COMMON_CANDIDATE_FEATURE_NAMES,
    build_bus_charging_decision_surface,
    build_bus_loading_decision_surface,
)
from rlaif.feature_alignment import align_named_features
from rlaif.reward_model_dataset import build_reward_pair_dataset
from tests.bus_event_chain_helpers import make_env
from tests.preference_v3_fixtures import rec


EXPECTED_LOADING_SCHEMA = (
    "time_norm",
    "freight_load_norm",
    "capacity_remaining_norm",
    "terminal_ready_parcel_count_norm",
    "onboard_passengers_norm",
    "soc_norm",
    "target_station_profile_norm",
)
EXPECTED_CHARGING_SCHEMA = (
    "time_norm",
    "soc_norm",
    "safety_energy_margin_norm",
    "current_delay_norm",
    "unloading_time_norm",
    "passenger_load_norm",
    "stop_queue_norm",
    "station_power_margin_norm",
)
EXPECTED_CANONICAL_SCHEMA = (
    "time_norm",
    "freight_load_norm",
    "capacity_remaining_norm",
    "terminal_ready_parcel_count_norm",
    "onboard_passengers_norm",
    "soc_norm",
    "target_station_profile_norm",
    "safety_energy_margin_norm",
    "current_delay_norm",
    "unloading_time_norm",
    "passenger_load_norm",
    "stop_queue_norm",
    "station_power_margin_norm",
)


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


def _actual_bus_surfaces(tmp_path):
    env = make_env(tmp_path)
    obs, _ = env.reset()
    loading = build_bus_loading_decision_surface(env, env.current_decision.event.payload["trip_id"])
    charging = None
    while obs["agent"] != "terminal":
        if obs.get("event_type_detail") == "BUS_STATION_ARRIVAL":
            charging = build_bus_charging_decision_surface(env, env.current_decision.event)
            break
        obs, *_ = env.step(first_feasible_policy(obs))
    assert charging is not None
    return loading, charging


def test_actual_bus_surface_schemas_match_source_constants_and_event_mapping(tmp_path):
    loading, charging = _actual_bus_surfaces(tmp_path)
    assert tuple(loading.feature_names) == BUS_LOADING_FEATURE_NAMES == EXPECTED_LOADING_SCHEMA
    assert tuple(charging.feature_names) == BUS_CHARGING_FEATURE_NAMES == EXPECTED_CHARGING_SCHEMA
    assert len(loading.feature_names) == 7
    assert len(charging.feature_names) == 8
    for event_type, surface in (
        ("BUS_TERMINAL_DEPARTURE", loading),
        ("BUS_STATION_ARRIVAL", charging),
    ):
        assert tuple(surface.feature_names) == BUS_EVENT_STATE_FEATURE_NAMES[event_type]


def test_bus_mixed_event_specific_schemas_succeed_after_canonicalization():
    ds = build_reward_pair_dataset([
        bus_rec("BUS_TERMINAL_DEPARTURE"),
        bus_rec("BUS_STATION_ARRIVAL"),
    ], agent_type="bus", formal_mode=True, require_bus_event_coverage=True)
    assert len(ds) == 2
    assert ds.state_feature_names == CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES
    assert ds.candidate_feature_names == COMMON_CANDIDATE_FEATURE_NAMES


def test_terminal_departure_values_align_to_canonical_positions_and_zero_fill():
    vals = [float(i + 1) for i in range(len(BUS_LOADING_FEATURE_NAMES))]
    aligned = align_named_features(BUS_LOADING_FEATURE_NAMES, vals, CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES)
    for name, value in zip(BUS_LOADING_FEATURE_NAMES, vals):
        assert aligned[CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES.index(name)] == value
    assert aligned[CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES.index("safety_energy_margin_norm")] == 0.0
    assert aligned[CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES.index("station_power_margin_norm")] == 0.0


def test_station_arrival_values_align_to_canonical_positions_and_zero_fill():
    vals = [float(i + 1) for i in range(len(BUS_CHARGING_FEATURE_NAMES))]
    aligned = align_named_features(BUS_CHARGING_FEATURE_NAMES, vals, CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES)
    for name, value in zip(BUS_CHARGING_FEATURE_NAMES, vals):
        assert aligned[CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES.index(name)] == value
    assert aligned[CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES.index("freight_load_norm")] == 0.0
    assert aligned[CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES.index("target_station_profile_norm")] == 0.0


def test_shared_fields_not_duplicated_and_order_is_deterministic():
    assert CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES == EXPECTED_CANONICAL_SCHEMA
    assert len(CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES) == 13
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
