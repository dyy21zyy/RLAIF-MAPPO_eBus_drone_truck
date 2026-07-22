from rlaif.runtime_agent_reward_model import RuntimeAgentRewardModel
from tests.rlaif_runtime_test_utils import make_ckpt

def test_checkpoint_input_normalization_changes_score(tmp_path):
    p1=make_ckpt(tmp_path/'a.pt', state_mean=[0,0], cand_mean=[0,0])
    p2=make_ckpt(tmp_path/'b.pt', state_mean=[10,10], cand_mean=[10,10])
    a=RuntimeAgentRewardModel.from_checkpoint(p1, expected_agent_type='assignment', formal_mode=False)
    b=RuntimeAgentRewardModel.from_checkpoint(p2, expected_agent_type='assignment', formal_mode=False)
    assert a.score(state_features=[1,2],candidate_features=[3,4],event_type='PARCEL_RELEASE').raw_score != b.score(state_features=[1,2],candidate_features=[3,4],event_type='PARCEL_RELEASE').raw_score
