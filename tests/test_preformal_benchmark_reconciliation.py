import pytest
from evaluation.preformal_part3_gates import REQUIRED_METRICS, BenchmarkIntegrityError, validate_reconciliation

def row(**over):
    m={k:{'value':0,'available':True,'finite':True,'source':'runtime','legitimate_zero':True} for k in REQUIRED_METRICS}
    m.update({'released_parcels':{'value':3},'delivered_parcels':{'value':2},'undelivered_parcels':{'value':1},'on_time_delivered_parcels':{'value':1},'urgent_parcels_released':{'value':1},'urgent_parcels_delivered_on_time':{'value':1},'truck_weight_utilization':{'value':.5},'truck_volume_utilization':{'value':.5},'bus_freight_utilization':{'value':.5},'charging_slot_utilization':{'value':.5},'locker_occupancy':{'value':.5},'environment_reward':{'value':2},'assignment_rlaif_contribution':{'value':1},'total_weighted_rlaif_reward':{'value':1},'combined_reward':{'value':3},'reward_fallback_count':{'value':0}})
    for k,v in over.items(): m[k]={'value':v}
    return {'method_id':'mappo_rlaif_assignment','formal_metrics':m}

def test_reconciles_and_validates_bounds_and_fallback():
    validate_reconciliation(row())
    with pytest.raises(BenchmarkIntegrityError): validate_reconciliation(row(truck_weight_utilization=2))
    with pytest.raises(BenchmarkIntegrityError): validate_reconciliation(row(reward_fallback_count=1))
    with pytest.raises(BenchmarkIntegrityError): validate_reconciliation(row(truck_rlaif_contribution=1,total_weighted_rlaif_reward=2,combined_reward=4))
