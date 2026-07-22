import pytest
from rlaif.reward_model_dataset import build_reward_pair_dataset
from tests.preference_v3_fixtures import rec

def test_wrong_feature_order_fails():
    with pytest.raises(ValueError): build_reward_pair_dataset([rec(candidate_b_feature_names=['d','c'])],agent_type='assignment')
def test_wrong_dimensions_fail():
    with pytest.raises(ValueError): build_reward_pair_dataset([rec(candidate_a_features=[1])],agent_type='assignment')
def test_wrong_schema_version_fails():
    with pytest.raises(ValueError): build_reward_pair_dataset([rec(event_schema_version=999)],agent_type='assignment')
