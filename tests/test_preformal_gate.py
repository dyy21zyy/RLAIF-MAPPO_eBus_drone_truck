from evaluation.preformal_gate import PREFORMAL_STAGES, STAGE_STATUSES, PreformalGate

def base(tmp_path, mode='diagnostic'):
    return {'run_classification':mode,'experiment_stage':'preformal_diagnostic' if mode=='diagnostic' else 'preformal','publication_eligible':False,'output_root':str(tmp_path/'results/diagnostic/gate' if mode=='diagnostic' else tmp_path/'results/preformal/gate')}

def test_canonical_stage_manifest_and_status_contract(tmp_path):
    g=PreformalGate(base(tmp_path))
    assert PREFORMAL_STAGES[0]=='repository_verification'
    assert PREFORMAL_STAGES[-1]=='formal_launch_plan'
    assert set(r.status for r in g.records.values()) == {'not_started'}
    assert 'blocked_dependency' in STAGE_STATUSES

def test_dependency_blocks_downstream_after_required_failure(tmp_path):
    def fail(_): raise RuntimeError('boom')
    report=PreformalGate(base(tmp_path), stage_functions={'scenario_bank_preparation':fail}).run()
    statuses={r['stage_id']:r['status'] for r in report['stages']}
    assert statuses['scenario_bank_preparation']=='failed'
    assert statuses['scenario_bank_validation']=='blocked_dependency'
    assert report['overall_status']=='PREFORMAL_DIAGNOSTIC_FAILED'

def test_diagnostic_success_uses_diagnostic_status_not_strict_success(tmp_path):
    report=PreformalGate(base(tmp_path)).run()
    assert report['overall_status']=='PREFORMAL_DIAGNOSTIC_PASSED'
    assert report['overall_status']!='PREFORMAL_ALL_REQUIRED_PATHS_PASSED'
    assert report['publication_eligible'] is False

def test_strict_rejects_diagnostic_artifact(tmp_path):
    artifact=tmp_path/'artifact.json'; artifact.write_text('{"run_classification":"diagnostic"}')
    cfg=base(tmp_path,'formal'); cfg['formal_candidate_artifacts']={'reward_model_validation':[str(artifact)]}
    report=PreformalGate(cfg).run()
    statuses={r['stage_id']:r['status'] for r in report['stages']}
    assert statuses['reward_model_validation']=='blocked_invalid_artifact'
    assert report['overall_status']=='BLOCKED_REWARD_MODELS'
