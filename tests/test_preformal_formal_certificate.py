from pathlib import Path
from evaluation.preformal_gate import PreformalGate, PREFORMAL_STAGES

def test_diagnostic_pass_cannot_issue_formal_certificate(tmp_path):
    cfg={'run_classification':'diagnostic','experiment_stage':'preformal_diagnostic','publication_eligible':False,'output_root':str(tmp_path/'results/diagnostic/g')}
    funcs={s:(lambda rec: None) for s in PREFORMAL_STAGES}
    report=PreformalGate(cfg,stage_functions=funcs).run()
    assert report['overall_status']=='PREFORMAL_DIAGNOSTIC_PASSED'
    assert not (tmp_path/'results/diagnostic/g/formal_experiment_readiness_certificate.json').exists()
