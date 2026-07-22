import pytest, torch
from rlaif.multi_agent_reward_model import AgentRewardModel
from training.event_schema import EVENT_NAME_TO_ID, validate_agent_event
def test_score_changes_with_event_id():
 m=AgentRewardModel(state_dim=1,candidate_dim=1,num_event_types=5,event_embedding_dim=2,hidden_dims=(4,),dropout=0); s=torch.zeros(1,1); c=torch.zeros(1,1); assert not torch.allclose(m(s,torch.tensor([0]),c),m(s,torch.tensor([1]),c))
def test_bus_accepts_both_and_rejects_unrelated():
 assert validate_agent_event('bus','BUS_TERMINAL_DEPARTURE'); assert validate_agent_event('bus','BUS_STATION_ARRIVAL')
 with pytest.raises(ValueError): validate_agent_event('bus','PARCEL_RELEASE')
def test_assignment_rejects_bus_events():
 with pytest.raises(ValueError): validate_agent_event('assignment','BUS_STATION_ARRIVAL')
