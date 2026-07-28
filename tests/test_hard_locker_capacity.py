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
