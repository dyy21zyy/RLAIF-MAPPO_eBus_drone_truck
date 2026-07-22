import pytest
from rlaif.reward_model_dataset import build_reward_pair_dataset
from tests.preference_v3_fixtures import rec

def test_exact_duplicate_deduplicated():
    ds=build_reward_pair_dataset([rec(), rec(preference_id='p2')],agent_type='assignment'); assert len(ds)==1 and ds.report.duplicate_count==1
def test_contradictory_duplicate_excluded():
    ds=build_reward_pair_dataset([rec(), rec(preference_id='p2', original_outcome='candidate_b')],agent_type='assignment'); assert len(ds)==0 and ds.report.contradiction_count==2
def test_self_comparison_fails():
    with pytest.raises(ValueError): build_reward_pair_dataset([rec(original_candidate_b_id='a', displayed_second_candidate_id='a')],agent_type='assignment')
def test_inconsistent_candidate_features_fail():
    with pytest.raises(ValueError): build_reward_pair_dataset([rec(), rec(preference_id='p2', candidate_a_features=[9,9])],agent_type='assignment')
