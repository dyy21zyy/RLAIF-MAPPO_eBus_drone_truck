from evaluation.policies import TruckDirectHeuristicPolicy, IntegratedRuleBasedPolicy, MAPPOPolicy
from training.event_schema import EVENT_NAME_TO_ID

def obs(agent='assignment'):
    return {'agent_id':agent,'event_type':'PARCEL_RELEASE','action_mask':[True,True,False],'candidate_actions':[{'action_id':0,'feasible':True,'mode':'TD','action_type':'truck_direct'},{'action_id':1,'feasible':True,'mode':'TBD','action_type':'truck_bus_drone'},{'action_id':2,'feasible':False}], 'candidate_features':[[0],[1],[2]]}

def test_truck_direct_selects_td_when_feasible():
    assert TruckDirectHeuristicPolicy().select_action(observation=obs(), env=None, deterministic=True) == 0

def test_integrated_selects_feasible_actions():
    assert IntegratedRuleBasedPolicy().select_action(observation=obs(), env=None, deterministic=True) in {0,1}

def test_mappo_receives_canonical_event_ids():
    p=MAPPOPolicy(); assert p.select_action(observation=obs(), env=None, deterministic=True)==0; assert p.event_ids_seen[-1] == EVENT_NAME_TO_ID['PARCEL_RELEASE']
