from experiments.run_paper_benchmark import should_skip_existing
def test_resume_identity():
    rows=[{'status':'success','method_id':'m','training_seed':1,'scenario_id':'s','policy_checkpoint_hash':'p','instance_hash':'i','resolved_evaluation_config_hash':'e'}]
    assert should_skip_existing(rows,method_id='m',training_seed=1,scenario_id='s',policy_hash='p',scenario_hash='i',evaluation_config_hash='e')
    assert not should_skip_existing(rows,method_id='m',training_seed=1,scenario_id='s',policy_hash='P2',scenario_hash='i',evaluation_config_hash='e')
    rows[0]['status']='failed'; assert not should_skip_existing(rows,method_id='m',training_seed=1,scenario_id='s',policy_hash='p',scenario_hash='i',evaluation_config_hash='e')
