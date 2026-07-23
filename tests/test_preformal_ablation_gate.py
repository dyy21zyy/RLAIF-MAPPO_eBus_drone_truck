from pathlib import Path
from evaluation.preformal_part3_gates import invoke_production_ablation

def test_production_ablation_runner_invoked(monkeypatch,tmp_path):
    monkeypatch.setattr('subprocess.run', lambda *a,**k: type('P',(),{'returncode':0,'stdout':'','stderr':''})())
    cfg=tmp_path/'c.yaml'; cfg.write_text('x: 1')
    assert 'experiments.run_ablation_matrix' in invoke_production_ablation(cfg,tmp_path/'out')['command']
