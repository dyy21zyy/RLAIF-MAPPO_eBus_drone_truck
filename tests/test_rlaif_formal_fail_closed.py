import pytest
from rlaif.reward_registry import RewardRegistry

def test_formal_cannot_enable_fallback():
    with pytest.raises(ValueError): RewardRegistry({"rlaif":{"enabled":True,"fallback_to_env_reward":True,"fail_on_invalid_reward_model":True,"agents":{}}})
