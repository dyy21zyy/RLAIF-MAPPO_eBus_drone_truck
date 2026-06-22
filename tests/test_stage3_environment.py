"""Stage 3 event-driven MDP regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_pipeline.build_instance import build_instance
from envs import DynamicDeliveryEnv, InstanceValidationError, first_feasible_policy
from experiments.smoke_test_environment import run_smoke_test

CONFIG = Path(__file__).parents[1] / "configs/shanghai_small.yaml"


@pytest.fixture()
def environment(tmp_path: Path) -> DynamicDeliveryEnv:
    instance = build_instance(CONFIG, fallback=True, output_root=tmp_path)
    return DynamicDeliveryEnv(Path(instance["output_directory"]) / "instance.json")


def advance_to_agent(env: DynamicDeliveryEnv, agent: str) -> dict[str, object]:
    observation, _ = env.reset()
    while observation["agent"] not in {agent, "terminal"}:
        observation, *_ = env.step(first_feasible_policy(observation))
    return observation


def test_reset_exposes_stable_assignment_schema(environment: DynamicDeliveryEnv) -> None:
    observation, info = environment.reset(seed=7)

    assert observation["agent"] == "assignment"
    assert observation["agent_id"] == "assignment"
    assert observation["event_type"] == "PARCEL_ARRIVAL"
    assert len(observation["features"]) == 17 + 10 * len(environment.station_ids)
    assert len(environment.get_global_state()) == 15
    assert len(observation["action_mask"]) == environment.assignment_action_size
    assert environment.assignment_action_size == 1 + 2 * len(environment.station_ids)
    assert observation["action_mask"][0]
    assert info["total_parcels"] == 60
    assert "reward_components" in info
    assert set(info["metrics"]) >= {"decision_events", "assignment_events", "bus_charging_events", "delivered_parcels", "undelivered_parcels", "drone_deliveries", "total_reward", "infeasible_action_corrections"}
    assert environment.config["bus"]["charging_actions_sec"] == [0, 15, 30, 45, 60, 75, 90, 105, 120]
    assert environment.check_invariants() == []


def test_infeasible_assignment_is_corrected_and_penalized(environment: DynamicDeliveryEnv) -> None:
    observation, _ = environment.reset()
    while all(observation["action_mask"]):
        observation, *_ = environment.step(0)
    invalid_action = observation["action_mask"].index(False)

    _observation, reward, _terminated, _truncated, info = environment.step(invalid_action)

    assert info["action_corrected"] is True
    assert info["applied_action"] != invalid_action
    assert reward <= -float(environment.config["reward"]["infeasible_action"])
    assert environment.cost_components["infeasible_action"] > 0
    assert environment.check_invariants() == []


def _current_parcel(environment: DynamicDeliveryEnv):
    assert environment.current_decision is not None
    return environment.parcels[environment.current_decision.event.payload["parcel_id"]]


def _make_station_drone_reachable(
    environment: DynamicDeliveryEnv, parcel_id: str, station_id: str
) -> None:
    environment.drone_distance_m[
        environment.drone_row_index[station_id], environment.drone_column_index[parcel_id]
    ] = 0.0


def test_heavy_parcel_masks_all_drone_paths(environment: DynamicDeliveryEnv) -> None:
    environment.reset()
    parcel = _current_parcel(environment)
    parcel.drone_feasible = True
    parcel.weight_kg = float(environment.config["network"]["drone_payload_kg"]) + 0.1

    mask = environment._assignment_mask(parcel)

    assert mask[0] is True
    assert not any(mask[1:])


def test_tld_is_masked_when_locker_remaining_capacity_is_insufficient(
    environment: DynamicDeliveryEnv,
) -> None:
    environment.reset()
    parcel = _current_parcel(environment)
    parcel.drone_feasible = True
    parcel.weight_kg = 1.0
    station_id = environment.station_ids[0]
    station = environment.stations[station_id]
    _make_station_drone_reachable(environment, parcel.parcel_id, station_id)
    station.locker_load_kg = station.locker_capacity_kg - parcel.weight_kg + 0.01

    mask = environment._assignment_mask(parcel)

    tld_action = 1 + len(environment.station_ids)
    assert mask[tld_action] is False


def test_tbd_is_masked_when_no_feasible_freight_bus_exists(
    environment: DynamicDeliveryEnv,
) -> None:
    environment.reset()
    parcel = _current_parcel(environment)
    parcel.drone_feasible = True
    parcel.weight_kg = 1.0
    station_id = environment.station_ids[0]
    _make_station_drone_reachable(environment, parcel.parcel_id, station_id)
    for trip in environment.trip_rows.values():
        trip["freight_allowed"] = "False"

    mask = environment._assignment_mask(parcel)

    assert mask[1] is False


def test_tbd_is_masked_when_bus_arrival_misses_parcel_deadline(
    environment: DynamicDeliveryEnv,
) -> None:
    environment.reset()
    parcel = _current_parcel(environment)
    parcel.drone_feasible = True
    parcel.weight_kg = 1.0
    parcel.deadline_min = environment.now_min + 1.0
    station_id = environment.station_ids[0]
    _make_station_drone_reachable(environment, parcel.parcel_id, station_id)

    mask = environment._assignment_mask(parcel)

    assert mask[1] is False


def test_tld_is_masked_when_truck_arrival_exceeds_delivery_horizon(
    environment: DynamicDeliveryEnv,
) -> None:
    environment.reset()
    parcel = _current_parcel(environment)
    parcel.drone_feasible = True
    parcel.weight_kg = 1.0
    station_id = environment.station_ids[0]
    _make_station_drone_reachable(environment, parcel.parcel_id, station_id)
    depot = environment.truck_location_index["depot_01"]
    station = environment.truck_location_index[station_id]
    environment.truck_time_min[depot, station] = environment.horizon_min + 1.0

    mask = environment._assignment_mask(parcel)

    tld_action = 1 + len(environment.station_ids)
    assert mask[tld_action] is False


def test_all_infeasible_actions_use_penalized_td_fallback(
    environment: DynamicDeliveryEnv,
) -> None:
    environment.config["truck"]["num_trucks"] = 0
    environment.parcel_rows[0]["weight"] = str(
        float(environment.config["truck"]["capacity_kg"]) + 1.0
    )
    observation, reset_info = environment.reset()

    assert observation["action_mask"] == [True] + [False] * (environment.assignment_action_size - 1)
    assert reset_info["metrics"]["fallback_feasibility_events"] == 1

    _next, reward, _terminated, _truncated, info = environment.step(0)

    assert info["action_corrected"] is True
    assert reward <= -float(environment.config["reward"]["infeasible_action"])
    assert info["metrics"]["fallback_feasibility_events"] >= 1
    assert environment.cost_components["infeasible_action"] > 0


def test_bus_decision_uses_configured_charging_actions(environment: DynamicDeliveryEnv) -> None:
    observation = advance_to_agent(environment, "bus")

    assert observation["agent"] == "bus"
    assert len(observation["features"]) == 6
    assert len(observation["action_mask"]) == environment.bus_action_size
    assert observation["action_mask"][0]
    trip_id = str(observation["entity_id"]).split(":", 1)[0]
    soc_before = environment.bus_soc_kwh[trip_id]
    charge_action = next((i for i, feasible in enumerate(observation["action_mask"]) if i > 0 and feasible), 0)
    _next, reward, *_ = environment.step(charge_action)

    assert environment.bus_soc_kwh[trip_id] >= soc_before
    assert reward <= 0
    assert environment.check_invariants() == []


def test_stage3_offline_smoke_completes_episode(tmp_path: Path) -> None:
    result = run_smoke_test(CONFIG, output_root=tmp_path)

    assert result["decisions"]["assignment"] == result["total_parcels"]
    assert 0 < result["delivered_parcels"] <= result["total_parcels"]
    assert result["decisions"]["bus"] > 0
    assert result["invariants"] == "passed"


def test_rejects_non_stage2_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "instance.json"
    manifest.write_text(json.dumps({"stage": 1}), encoding="utf-8")

    with pytest.raises(InstanceValidationError, match="stage: 2"):
        DynamicDeliveryEnv(manifest)


def test_station_power_overload_is_soft_penalty(environment: DynamicDeliveryEnv) -> None:
    observation = advance_to_agent(environment, "bus")
    station_id = str(observation["entity_id"]).split(":", 1)[1]
    station = environment.stations[station_id]
    station.power_capacity_kw = 1.0
    mask = environment._bus_mask(environment.current_decision.event)

    assert mask[1] is True
    _next, reward, *_rest, info = environment.step(1)
    assert reward < 0
    assert info["cost_components"]["power_overload"] > 0


def test_power_overload_is_integrated_over_elapsed_time(
    environment: DynamicDeliveryEnv,
) -> None:
    environment.reset()
    station = environment.stations[environment.station_ids[0]]
    base_load = float(environment.config["station"]["base_load_kw"])
    bus_load = float(environment.config["bus"]["charging_power_kw"])
    station.power_capacity_kw = base_load + 20.0
    station.active_bus_charges = [10.0]

    environment._integrate_station_penalties(0.0, 10.0)

    assert environment.accumulated_power_overload == pytest.approx((bus_load - 20.0) * 10.0)


def test_power_overload_duration_only_counts_positive_intervals(
    environment: DynamicDeliveryEnv,
) -> None:
    environment.reset()
    station = environment.stations[environment.station_ids[0]]
    station.power_capacity_kw = float(environment.config["station"]["base_load_kw"])
    environment._integrate_station_penalties(0.0, 4.0)
    station.active_bus_charges = [10.0]
    environment._integrate_station_penalties(4.0, 10.0)

    assert environment.accumulated_power_overload_duration == pytest.approx(6.0)


def test_locker_overflow_is_integrated_over_elapsed_time(
    environment: DynamicDeliveryEnv,
) -> None:
    environment.reset()
    station = environment.stations[environment.station_ids[0]]
    station.locker_load_kg = station.locker_capacity_kg + 2.0

    environment._integrate_station_penalties(0.0, 5.0)

    assert environment.accumulated_locker_overflow == pytest.approx(10.0)


def test_locker_overflow_duration_only_counts_positive_intervals(
    environment: DynamicDeliveryEnv,
) -> None:
    environment.reset()
    station = environment.stations[environment.station_ids[0]]
    environment._integrate_station_penalties(0.0, 3.0)
    station.locker_load_kg = station.locker_capacity_kg + 1.0
    environment._integrate_station_penalties(3.0, 8.0)

    assert environment.accumulated_locker_overflow_duration == pytest.approx(5.0)


def test_drone_dispatch_never_creates_negative_locker_load(
    environment: DynamicDeliveryEnv,
) -> None:
    environment.reset()
    parcel = _current_parcel(environment)
    parcel.weight_kg = 1.0
    parcel.drone_feasible = True
    station_id = environment.station_ids[0]
    _make_station_drone_reachable(environment, parcel.parcel_id, station_id)

    environment._handle_station_arrival(parcel.parcel_id, station_id)

    assert environment.stations[station_id].locker_load_kg >= 0.0


def test_locker_load_persists_until_delayed_drone_dispatch(
    environment: DynamicDeliveryEnv,
) -> None:
    environment.reset()
    parcel = _current_parcel(environment)
    parcel.weight_kg = 1.0
    parcel.drone_feasible = True
    station_id = environment.station_ids[0]
    station = environment.stations[station_id]
    _make_station_drone_reachable(environment, parcel.parcel_id, station_id)
    station.full_batteries = 0
    station.battery_ready_min = [environment.now_min + 5.0]

    environment._handle_station_arrival(parcel.parcel_id, station_id)

    assert station.locker_load_kg == pytest.approx(parcel.weight_kg)
    assert any(event.kind == "drone_dispatch" for event in environment.events)


def test_metrics_expose_station_penalty_amounts_and_durations(
    environment: DynamicDeliveryEnv,
) -> None:
    environment.reset()
    station = environment.stations[environment.station_ids[0]]
    station.power_capacity_kw = float(environment.config["station"]["base_load_kw"])
    station.active_bus_charges = [2.0]
    station.locker_load_kg = station.locker_capacity_kg + 1.0
    environment._integrate_station_penalties(0.0, 2.0)

    metrics = environment.get_metrics()

    assert metrics["power_overload_amount"] > 0.0
    assert metrics["power_overload_duration"] == pytest.approx(2.0)
    assert metrics["locker_overflow_amount"] == pytest.approx(2.0)
    assert metrics["locker_overflow_duration"] == pytest.approx(2.0)


def test_station_drone_cycle_preserves_non_negative_resources(environment: DynamicDeliveryEnv) -> None:
    first_parcel = environment.parcel_rows[0]
    first_parcel["weight"] = "1.0"
    first_parcel["drone_feasible"] = "True"
    _make_station_drone_reachable(environment, first_parcel["parcel_id"], environment.station_ids[0])
    observation, _ = environment.reset()
    first_tld_action = 1 + len(environment.station_ids)
    while observation["agent"] != "terminal":
        feasible_tld = (
            next(
                (
                    action_id
                    for action_id in range(first_tld_action, environment.assignment_action_size)
                    if observation["action_mask"][action_id]
                ),
                None,
            )
            if observation["agent"] == "assignment"
            else None
        )
        if feasible_tld is not None:
            action = feasible_tld
        else:
            action = first_feasible_policy(observation)
        observation, *_ = environment.step(action)

    assert any(parcel.mode == "TLD" for parcel in environment.parcels.values())
    assert all(station.locker_load_kg >= 0 for station in environment.stations.values())
    assert all(station.full_batteries >= 0 for station in environment.stations.values())
    assert environment.check_invariants() == []
