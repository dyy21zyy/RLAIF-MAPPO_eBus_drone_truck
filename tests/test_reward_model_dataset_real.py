from rlaif.reward_model_dataset import build_reward_pair_dataset
from tests.preference_v3_fixtures import rec

def test_binary_rows_enter_dataset():
    ds=build_reward_pair_dataset([rec()],agent_type='assignment'); assert len(ds)==1 and ds[0].preference_id=='p1'
def test_ties_abstentions_invalid_unresolved_excluded_and_reported():
    rows=[rec(preference_id=f'p{i}', original_outcome=o, original_candidate_a_id=f'a{i}', original_candidate_b_id=f'b{i}', displayed_first_candidate_id=f'a{i}', displayed_second_candidate_id=f'b{i}') for i,o in enumerate(['tie','abstain','invalid','unresolved'])]
    ds=build_reward_pair_dataset(rows,agent_type='assignment'); assert len(ds)==0; assert ds.report.excluded_outcomes['tie']==1 and ds.report.excluded_outcomes['abstain']==1 and ds.report.excluded_outcomes['invalid']==1 and ds.report.excluded_outcomes['unresolved']==1
