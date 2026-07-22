import pytest
from training.reward_contribution import build_reward_contribution

def test_deterministic_rlaif_order():
    c=build_reward_contribution(agent_type="assignment",event_type="PARCEL_RELEASE",environment_reward=-1.0,raw_learned_reward=5.0,mean=1.0,std=2.0,epsilon=0.0,clip_bound=1.5,lambda_value=0.2)
    assert c.normalized_learned_reward==pytest.approx(2.0)
    assert c.clipped_learned_reward==pytest.approx(1.5)
    assert c.weighted_learned_contribution==pytest.approx(0.3)
    assert c.total_reward==pytest.approx(-0.7)
