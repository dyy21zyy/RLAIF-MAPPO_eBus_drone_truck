from pathlib import Path
from evaluation.preformal_part3_gates import invoke_production_benchmark

def test_invokes_production_benchmark_runner(monkeypatch,tmp_path):
    calls=[]
    def fake(cmd, cwd, text, capture_output, check):
        calls.append(cmd)
        class P: returncode=0; stdout='ok'; stderr=''
        return P()
    monkeypatch.setattr('subprocess.run', fake)
    cfg=tmp_path/'c.yaml'; cfg.write_text('x: 1')
    rec=invoke_production_benchmark(cfg,tmp_path/'out')
    assert 'experiments.run_paper_benchmark' in rec['command'] and rec['return_code']==0
