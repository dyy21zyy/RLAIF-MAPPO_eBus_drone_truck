from types import SimpleNamespace

import pytest

import evaluation.formal_metrics
from evaluation.metrics import FormalMetricError
from experiments.run_fixed_policy_sensitivity import AGENTS, IDENTITY_FIELDS, _flat_metrics, evaluation_identity, reject_duplicate_identities, should_skip

def row(**updates):
    r={k:f"x-{k}" for k in IDENTITY_FIELDS}; r.update(status="success",fallback_count=0); r.update(updates); return r

def test_resume_identity_is_hash_aware_and_success_only():
    good=row(); assert should_skip([good],evaluation_identity(good))
    assert not should_skip([row(status="failed")],evaluation_identity(good))
    assert not should_skip([row(policy_checkpoint_hash="changed")],evaluation_identity(good))

def test_duplicate_identity_rejected():
    with pytest.raises(ValueError,match="duplicated"): reject_duplicate_identities([row(),row()])

def test_reward_models_cannot_enter_resume_or_action_identity():
    # Action selection identity is policy/scenario only; reward models are immutable audit lineage.
    assert "reward_model_score" not in IDENTITY_FIELDS

def test_obsolete_bus_arrival_identifier_absent_from_runner_source():
    source=open("experiments/run_fixed_policy_sensitivity.py").read()
    assert '"BUS_ARRIVAL"' not in source

def test_fallback_rows_are_not_successful_contract():
    with pytest.raises(ValueError,match="fallback"):
        from experiments.aggregate_fixed_policy_sensitivity import validate_rows
        validate_rows([row(sensitivity_family="passenger",parameter_value="1",policy_seed="1",paired_scenario_index="0",scenario_id="s",scenario_content_hash="h",sensitivity_mode="fixed_policy_robustness",run_classification="formal",policy_checkpoint_hash="p",scenario_bank_hash="b",fallback_count="1")],expected_scenarios=1,expected_seeds=(1,))

def formal_env():
    parcels={
        1: SimpleNamespace(status="DELIVERED", delivered_time_min=15, deadline_min=10, release_time_min=0, is_urgent=True),
        2: SimpleNamespace(status="DELIVERED", delivered_time_min=8, deadline_min=10, release_time_min=0, is_urgent=False),
        3: SimpleNamespace(status="PENDING", delivered_time_min=None, deadline_min=20, release_time_min=0, is_urgent=False),
    }
    return SimpleNamespace(
        parcels=parcels, trucks=[SimpleNamespace(total_distance=12)],
        truck_dispatch_count=2, truck_weight_utilization_sum=1.0,
        truck_volume_utilization_sum=.8, truck_parcels_routed=4,
        bus_freight_utilization=.25, bus_propulsion_energy_kwh=3,
        bus_charging_energy_kwh=4, bus_soc_kwh={"bus": 5},
        battery_safety_violation_count=1, passenger_waiting_minutes=12,
        passenger_onboard_delay_minutes=6,
        raw_cost_components={"bus_operating_delay": 7}, drone_mission_count=2,
        charging_slot_busy_minutes=5, charging_slot_available_minutes=10,
        locker_occupancy_kg_minutes=9, peak_station_load_kw=80,
        accumulated_power_overload=3, passenger_boardings_at_ordinary_stops=2,
        config={"station": {"power_capacity_kw": 100}},
    )


def audit(**updates):
    values={f"rlaif_reward_{agent}": 2.0 for agent in AGENTS}
    values.update({f"{agent}_decision_count": 2 for agent in AGENTS})
    values.update(fallback_count=0, total_reward_clipping_count=0)
    values.update(updates)
    return values


def test_successful_row_contains_canonical_metrics_and_actual_lateness():
    metrics=_flat_metrics(formal_env(), audit())
    required={"average_lateness", "maximum_lateness", "on_time_over_all_released",
              "on_time_over_delivered", "urgent_on_time_fulfillment",
              "truck_weight_utilization", "truck_volume_utilization",
              "parcels_per_truck_route", "bus_freight_utilization",
              "battery_safety_violations", "bus_operating_delay",
              "charging_slot_utilization", "locker_occupancy"}
    assert required <= metrics.keys()
    assert metrics["average_lateness"] == 2.5
    assert metrics["maximum_lateness"] == 5


def test_placeholder_collector_is_not_used(monkeypatch):
    monkeypatch.setattr(evaluation.formal_metrics, "collect_formal_metrics",
                        lambda env: (_ for _ in ()).throw(AssertionError("placeholder collector used")))
    assert _flat_metrics(formal_env(), audit())["average_lateness"] == 2.5


def test_missing_canonical_source_fails_closed():
    env=formal_env()
    del env.truck_weight_utilization_sum
    with pytest.raises(FormalMetricError, match="truck_weight_utilization_sum"):
        _flat_metrics(env, audit())


def test_passenger_denominators_and_reward_decision_conventions():
    metrics=_flat_metrics(formal_env(), audit(assignment_decision_count=0))
    assert metrics["total_boarded_passengers"] == 2
    assert metrics["waiting_minutes_per_passenger"] == 6
    assert metrics["onboard_delay_minutes_per_passenger"] == 3
    assert metrics["assignment_reward_per_decision"] is None
    assert metrics["truck_reward_per_decision"] == 1
    assert metrics["fallback_count"] == 0


def test_runner_calls_strict_canonical_collector():
    source=open("experiments/run_fixed_policy_sensitivity.py").read()
    assert "evaluation.metrics.collect_formal_runtime_metrics(env)" in source
    assert "evaluation.formal_metrics.collect_formal_metrics" not in source
