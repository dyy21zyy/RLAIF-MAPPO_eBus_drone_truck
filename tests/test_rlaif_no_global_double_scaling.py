import pytest
from training.reward_contribution import build_reward_contribution

def test_lambda_applied_once():
    c=build_reward_contribution(agent_type="bus",event_type="BUS_STATION_ARRIVAL",environment_reward=1,raw_learned_reward=2,mean=0,std=1,clip_bound=2,lambda_value=.2)
    assert c.weighted_learned_contribution==pytest.approx(.4)
    assert c.total_reward==pytest.approx(1.4)
