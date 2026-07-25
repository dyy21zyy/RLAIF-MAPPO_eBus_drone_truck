import pytest
from experiments.run_fixed_policy_sensitivity import evaluation_identity, reject_duplicate_identities, should_skip, IDENTITY_FIELDS

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

def test_passenger_metric_uses_explicit_environment_denominator():
    source=open("envs/delivery_env.py").read(); runner=open("experiments/run_fixed_policy_sensitivity.py").read()
    assert '"total_boarded_passengers"' in source and 'm.get("total_boarded_passengers"' in runner
    assert "parcel" not in runner[runner.index('boarded=m.get("total_boarded_passengers"'):runner.index('required={')]
