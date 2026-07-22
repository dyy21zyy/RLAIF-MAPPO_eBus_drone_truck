import torch
from rlaif.runtime_agent_reward_model import RuntimeAgentRewardModel
from tests.rlaif_runtime_test_utils import make_ckpt

def test_bus_events_use_global_event_ids(tmp_path):
    p=make_ckpt(tmp_path/'bus.pt','bus')
    m=RuntimeAgentRewardModel.from_checkpoint(p, expected_agent_type='bus', formal_mode=False)
    seen=[]
    old=m.model.forward
    def spy(state,event_ids,cand):
        seen.append(int(event_ids[0])); return old(state,event_ids,cand)
    m.model.forward=spy
    m.score(state_features=[0,0],candidate_features=[0,0],event_type='BUS_TERMINAL_DEPARTURE')
    m.score(state_features=[0,0],candidate_features=[0,0],event_type='BUS_STATION_ARRIVAL')
    assert seen == [2,3]
