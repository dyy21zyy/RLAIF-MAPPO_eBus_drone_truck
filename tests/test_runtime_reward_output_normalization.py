import pytest, torch
from rlaif.runtime_agent_reward_model import RuntimeAgentRewardModel
from training.reward_model_wrapper import RewardCheckpointCompatibilityError
from tests.rlaif_runtime_test_utils import make_ckpt

def test_output_uses_checkpoint_training_stats(tmp_path):
    p=make_ckpt(tmp_path/'a.pt', reward_mean=2.0, reward_std=2.0)
    m=RuntimeAgentRewardModel.from_checkpoint(p, expected_agent_type='assignment', formal_mode=False)
    s=m.score(state_features=[0,0],candidate_features=[0,0],event_type='PARCEL_RELEASE')
    assert s.normalized_score == pytest.approx((s.raw_score-2.0)/(2.0+1e-6))

def test_bad_output_std_rejected(tmp_path):
    p=make_ckpt(tmp_path/'a.pt'); ck=torch.load(p,weights_only=False); ck['reward_output_training_std']=0.0; torch.save(ck,p)
    with pytest.raises(RewardCheckpointCompatibilityError): RuntimeAgentRewardModel.from_checkpoint(p, expected_agent_type='assignment', formal_mode=False)
