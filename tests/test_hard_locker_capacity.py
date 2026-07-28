"""Focused contracts for inbound locker reservations and formal hard gates."""
from pathlib import Path

import pytest

from data_pipeline.build_instance import build_instance
from envs import DynamicDeliveryEnv
from evaluation.formal_metric_validation import FormalMetricValidationError, validate_formal_metrics


@pytest.fixture()
def env(tmp_path: Path) -> DynamicDeliveryEnv:
    instance = build_instance(Path("configs/shanghai_small.yaml"), fallback=True, output_root=tmp_path)
    environment = DynamicDeliveryEnv(Path(instance["output_directory"]) / "instance.json")
    environment.reset(seed=7)
    return environment


def current(env: DynamicDeliveryEnv):
    return next(iter(env.parcels.values()))


def test_reserve_cancel_is_exact_and_duplicate_raises(env):
    parcel = current(env); station = env.stations[env.station_ids[0]]
    assert env._reserve_locker_capacity(parcel.parcel_id, station.station_id)
    assert station.locker_reserved_kg == pytest.approx(parcel.weight_kg)
    assert station.locker_load_kg == 0
    with pytest.raises(RuntimeError, match="duplicate"):
        env._reserve_locker_capacity(parcel.parcel_id, station.station_id)
    assert env._cancel_locker_reservation(parcel.parcel_id, reason="test")
    assert station.locker_reserved_kg == 0


def test_insufficient_reservation_has_no_partial_mutation(env):
    parcel = current(env); station = env.stations[env.station_ids[0]]
    station.locker_load_kg = station.locker_capacity_kg
    assert not env._reserve_locker_capacity(parcel.parcel_id, station.station_id)
    assert station.locker_reserved_kg == 0 and not env.inbound_locker_reservations


def test_arrival_converts_reservation_to_occupancy_and_dispatch_releases(env):
    parcel = current(env); station = env.stations[env.station_ids[0]]
    parcel.station_id = station.station_id
    assert env._reserve_locker_capacity(parcel.parcel_id, station.station_id)
    env._handle_station_arrival(parcel.parcel_id, station.station_id)
    assert station.locker_reserved_kg == 0
    assert station.locker_load_kg == pytest.approx(parcel.weight_kg)
    env._release_locker_occupancy(parcel.parcel_id, station.station_id)
    parcel.status = "ONBOARD_DRONE"
    assert station.locker_load_kg == 0
    with pytest.raises(RuntimeError):
        env._release_locker_occupancy(parcel.parcel_id, station.station_id)


def test_arrival_requires_matching_reservation_and_weight(env):
    parcel = current(env); station = env.stations[env.station_ids[0]]
    with pytest.raises(RuntimeError, match="without"):
        env._handle_station_arrival(parcel.parcel_id, station.station_id)
    assert env._reserve_locker_capacity(parcel.parcel_id, station.station_id)
    parcel.weight_kg += 1
    with pytest.raises(RuntimeError, match="weight mismatch"):
        env._handle_station_arrival(parcel.parcel_id, station.station_id)


def test_reserved_load_masks_tld_and_tbd(env):
    parcel = current(env); station_id = env.station_ids[0]; station = env.stations[station_id]
    parcel.drone_feasible = True
    env.drone_distance_m[env.drone_row_index[station_id], env.drone_column_index[parcel.parcel_id]] = 0
    station.locker_reserved_kg = station.locker_capacity_kg
    mask = env._assignment_mask(parcel)
    assert mask[1] is False
    assert mask[1 + len(env.station_ids)] is False


def test_invariants_detect_effective_overflow_and_registry_mismatch(env):
    station = env.stations[env.station_ids[0]]
    station.locker_reserved_kg = station.locker_capacity_kg + 1
    errors = env.check_invariants()
    assert any("effective locker load exceeds" in error for error in errors)
    assert any("registry total mismatch" in error for error in errors)


def test_formal_hard_gate_rejects_positive_values():
    # Isolate the gate from the broader required metric schema.
    import evaluation.formal_metric_validation as validation
    required = validation.REQUIRED_FORMAL_METRICS
    row = {name: {"value": 0.0, "availability": "available", "source": "test", "legitimate_zero": True} for name in required + validation.RLAIF_FIELDS}
    row["locker_overflow_amount"]["value"] = 1.0
    with pytest.raises(FormalMetricValidationError, match="hard locker"):
        validate_formal_metrics(row)


def test_waiting_index_invariants(env):
    parcel = current(env); station = env.stations[env.station_ids[0]]
    parcel.station_id = station.station_id
    assert env._reserve_locker_capacity(parcel.parcel_id, station.station_id)
    env._handle_station_arrival(parcel.parcel_id, station.station_id)
    assert env.check_invariants() == []
    env.waiting_station_parcels[station.station_id].append(parcel.parcel_id)
    assert any("duplicated" in error for error in env.check_invariants())


def test_missing_and_wrong_waiting_index_are_rejected(env):
    parcel = current(env); station_id = env.station_ids[0]
    parcel.status, parcel.station_id = "WAITING_DRONE", station_id
    env.stations[station_id].locker_load_kg = parcel.weight_kg
    assert any("missing" in error for error in env.check_invariants())
    other = env.station_ids[1]
    env.waiting_station_parcels[other] = [parcel.parcel_id]
    assert any("wrong station" in error for error in env.check_invariants())


def test_legacy_td_cancels_and_tbd_reserves(env):
    parcel = current(env); station_id = env.station_ids[0]
    assert env._reserve_locker_capacity(parcel.parcel_id, station_id)
    env._apply_assignment(parcel.parcel_id, 0)
    assert parcel.parcel_id not in env.inbound_locker_reservations

    parcel2 = list(env.parcels.values())[1]
    env._apply_assignment(parcel2.parcel_id, 1)
    assert parcel2.parcel_id in env.inbound_locker_reservations
    assert any(task["parcel_id"] == parcel2.parcel_id for task in env.pending_truck_tasks)


def test_legacy_capacity_failure_is_terminal_and_atomic(env):
    parcel = current(env); station = env.stations[env.station_ids[0]]
    station.locker_load_kg = station.locker_capacity_kg
    before = env.infeasible_action_corrections
    env._apply_assignment(parcel.parcel_id, 1)
    assert parcel.status == "FAILED"
    assert parcel.parcel_id not in env.inbound_locker_reservations
    assert not any(task.get("parcel_id") == parcel.parcel_id for task in env.pending_truck_tasks)
    assert env.infeasible_action_corrections == before + 1


def test_action_time_capacity_failure_is_terminal_and_atomic(env, monkeypatch):
    parcel = current(env); station = env.stations[env.station_ids[0]]
    parcel.drone_feasible = True
    env.drone_distance_m[env.drone_row_index[station.station_id], env.drone_column_index[parcel.parcel_id]] = 0
    assert env._assignment_mask(parcel)[1 + len(env.station_ids)]
    station.locker_load_kg = station.locker_capacity_kg
    before = env.infeasible_action_corrections
    env._apply_assignment_decision(parcel.parcel_id, 1 + len(env.station_ids))
    assert parcel.status == "FAILED"
    assert parcel.parcel_id not in env.inbound_locker_reservations
    assert not any(task.get("parcel_id") == parcel.parcel_id for task in env.pending_truck_tasks)
    assert env.infeasible_action_corrections == before + 1
