import json, math, pytest
from pathlib import Path
from envs.reward_components import REWARD_COMPONENTS
from envs.reward_ledger import RewardLedger
from envs.reward_scales import load_reward_scale_artifact
from experiments.estimate_reward_reference_scales import percentile_scale, classify_and_estimate, InvalidRewardScaleScenarioBankError, validate_training_bank
from evaluation.reward_scale_reference_policies import truck_direct_reference, integrated_rule_reference, coverage_reference
from tests.helpers.diagnostic_reward_scale_artifacts import write_scale_artifact

def test_percentile_and_overrides(tmp_path):
    assert percentile_scale([1,2,100], percentile=50)==2
    with pytest.raises(Exception): percentile_scale([0,0])
    with pytest.raises(Exception): percentile_scale([1,float('nan')])
    rows=[{'episode_status':'success', **{f'raw_{c}':0.0 for c in REWARD_COMPONENTS}}]
    rows[0]['raw_truck_cost']=5
    comps, scales, stats=classify_and_estimate(rows, {'minimum_scale_overrides': {'bus_battery_violation': {'value': 1, 'reason': 'documented'}}})
    assert comps['truck_cost']['status']=='observed_positive' and scales['truck_cost']>0
    assert comps['bus_battery_violation']['status']=='instrumented_zero'

def test_artifact_hash_lineage_and_formal_rejection(tmp_path):
    p=tmp_path/'a.json'; payload=write_scale_artifact(p, bank_hash='bank')
    assert load_reward_scale_artifact(p, expected_hash=payload['artifact_hash'], expected_training_bank_hash='bank', formal_mode=True).scales['truck_cost']==1
    bad=json.loads(p.read_text()); bad['scales']['truck_cost']=2; p.write_text(json.dumps(bad))
    with pytest.raises(ValueError): load_reward_scale_artifact(p)
    d=tmp_path/'d.json'; write_scale_artifact(d, run_classification='diagnostic')
    with pytest.raises(ValueError): load_reward_scale_artifact(d, formal_mode=True)

def test_ledger_reconciles_scale_weight_reward():
    l=RewardLedger(); r=l.add_cost(event_time=0, component='truck_cost', raw_amount=10, weight=2, reference_scale=5, scale_artifact_hash='h')
    e=l.entries[0]
    assert e.raw_amount==10 and e.reference_scale==5 and e.normalized_amount==2 and e.weighted_amount==4 and e.reward_contribution==-4 and r==-4 and l.reward_sum()==-4

def test_reference_policies_are_deterministic_and_feasible():
    obs={'action_mask':[False, True, True], 'candidates':['bad','charge','dispatch']}
    for maker in [truck_direct_reference, integrated_rule_reference, coverage_reference]:
        p=maker(); a=p.select_action(obs); assert a in [1,2]; assert a==p.select_action(obs); assert p.version==1

def test_train_bank_validation_rejects_non_train(tmp_path):
    m=tmp_path/'scenario_bank_manifest.json'; m.write_text(json.dumps({'split':'validation','bank_hash':'h','scenarios':[]}))
    with pytest.raises(InvalidRewardScaleScenarioBankError): validate_training_bank(m)
