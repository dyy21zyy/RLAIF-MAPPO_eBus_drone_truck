import pytest
from training.reward_contribution import build_reward_contribution

def test_stage_formula_separation():
    c=build_reward_contribution(agent_type='truck',event_type='TRUCK_AVAILABLE',environment_reward=10,raw_learned_reward=5,mean=1,std=2,clip_bound=1,lambda_value=.5)
    assert c.normalized_learned_reward == pytest.approx(2.0, abs=1e-5)
    assert c.clipped_learned_reward == 1
    assert c.weighted_learned_contribution == .5
    assert c.total_reward == 10.5
