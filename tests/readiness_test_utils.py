import json, subprocess, sys
from pathlib import Path

def run_pilot(tmp_path):
    out=tmp_path/'pilot'
    subprocess.check_call([sys.executable,'-m','experiments.run_readiness_pilot','--config','configs/paper/train_mappo_env.yaml','--episodes','20','--seed','1','--collect-traces','--output',str(out),'--overwrite'])
    return out

def load(out,name): return json.loads((Path(out)/name).read_text())
