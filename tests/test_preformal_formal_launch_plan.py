from evaluation.formal_launch_plan import generate_formal_launch_plan, STAGES

def test_launch_plan_contains_final_matrix_and_executes_nothing(tmp_path):
    plan=generate_formal_launch_plan({'input_hashes':{'scenario_generation_config_hash':'h','scenario_bank_hash':'b','reward_scale_hash':'r','reward_model_hash':'m','benchmark_config_hash':'bc','ablation_config_hash':'ac','sensitivity_config_hash':'sc','result_integrity_config_hash':'ic'}}, tmp_path/'formal_launch_plan.json')
    assert set(STAGES) <= {s['stage_id'] for s in plan['stages']}
    assert plan['training_seeds']==[1,2,3] and plan['mappo_total_episodes']==3000 and plan['test_scenarios']==100
    assert plan['expected_benchmark_rows']['total']==(4*3*100)+(2*100)
    assert (tmp_path/'formal_launch_plan.json').exists()

def test_unresolved_placeholder_blocks(tmp_path):
    plan=generate_formal_launch_plan({'input_hashes':{'scenario_generation_config_hash':'REPLACE_WITH_REAL_HASH'}}, tmp_path/'p.json')
    assert any(s['status']=='blocked_missing_artifact' for s in plan['stages'])
