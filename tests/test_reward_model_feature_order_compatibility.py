import pytest
from rlaif.reward_model_dataset import build_reward_pair_dataset
from tests.preference_v3_fixtures import rec

def test_expected_state_order_mismatch_fails():
    with pytest.raises(ValueError): build_reward_pair_dataset([rec()],agent_type='assignment',expected_state_feature_names=('y','x'))
def test_expected_candidate_order_mismatch_fails():
    with pytest.raises(ValueError): build_reward_pair_dataset([rec()],agent_type='assignment',expected_candidate_feature_names=('d','c'))
