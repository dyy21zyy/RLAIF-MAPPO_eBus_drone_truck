from evaluation.policies import TruckDirectHeuristicPolicy, IntegratedRuleBasedPolicy, MAPPOPolicy

def obs(agent='assignment'):
    return {'agent_id':agent,'event_type':'PARCEL_RELEASE','action_mask':[True,True,False],'candidate_actions':[{'action_id':0,'feasible':True,'mode':'TD','action_type':'truck_direct'},{'action_id':1,'feasible':True,'mode':'TBD','action_type':'truck_bus_drone'},{'action_id':2,'feasible':False}], 'candidate_features':[[0],[1],[2]]}

def test_truck_direct_selects_td_when_feasible():
    assert TruckDirectHeuristicPolicy().select_action(observation=obs(), env=None, deterministic=True) == 0

def test_integrated_selects_feasible_actions():
    assert IntegratedRuleBasedPolicy().select_action(observation=obs(), env=None, deterministic=True) in {0,1}

def test_mappo_requires_checkpoint():
    import pytest

    with pytest.raises(
        ValueError,
        match="requires a checkpoint",
    ):
        MAPPOPolicy()
