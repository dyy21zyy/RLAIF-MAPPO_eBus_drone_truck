from evaluation.preformal_part3_gates import validate_expected_rows

def test_expected_row_count_derived():
    methods=[{'method_id':'truck_direct_heuristic'},{'method_id':'mappo_env','training_seeds':[1,2]}]
    rep=validate_expected_rows(methods,['s1','s2'],[{'method_id':'truck_direct_heuristic','training_seed':None,'scenario_id':'s1','status':'success'}])
    assert rep['expected_rows']==6 and rep['missing_rows']
