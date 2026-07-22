import pytest, torch
from rlaif.runtime_agent_reward_model import RuntimeAgentRewardModel
from training.reward_model_wrapper import RewardCheckpointCompatibilityError
from tests.rlaif_runtime_test_utils import make_ckpt

def test_feature_order_and_schema_mismatch_fail(tmp_path):
    p=make_ckpt(tmp_path/'a.pt')
    with pytest.raises(RewardCheckpointCompatibilityError): RuntimeAgentRewardModel.from_checkpoint(p, expected_agent_type='assignment', expected_state_feature_names=['s1','s0'], formal_mode=False)
    ck=torch.load(p,weights_only=False); ck['observation_schema_version']=999; torch.save(ck,p)
    with pytest.raises(RewardCheckpointCompatibilityError): RuntimeAgentRewardModel.from_checkpoint(p, expected_agent_type='assignment', formal_mode=False)
