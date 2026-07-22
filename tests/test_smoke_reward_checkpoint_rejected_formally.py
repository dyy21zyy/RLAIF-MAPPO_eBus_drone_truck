import pytest
from rlaif.runtime_agent_reward_model import RuntimeAgentRewardModel
from training.reward_model_wrapper import RewardCheckpointValidationError
from tests.rlaif_runtime_test_utils import make_ckpt

def test_smoke_rejected_formal(tmp_path):
    p=make_ckpt(tmp_path/'a.pt')
    with pytest.raises(RewardCheckpointValidationError): RuntimeAgentRewardModel.from_checkpoint(p, expected_agent_type='assignment', formal_mode=True)
