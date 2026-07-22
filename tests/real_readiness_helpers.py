from __future__ import annotations
import json, subprocess, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_CACHE = ROOT / "results" / "pytest_real_readiness_cache"
_RAN = False

def run_pilot(tmp_path=None, episodes=2):
    global _RAN
    out=_CACHE
    if not _RAN or not (out/"readiness_summary.json").exists():
        if out.exists(): shutil.rmtree(out)
        cmd=[sys.executable,"-m","experiments.run_readiness_pilot","--config","configs/diagnostic/readiness_pilot.yaml","--episodes",str(episodes),"--seed","1","--collect-traces","--output",str(out),"--overwrite"]
        res=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=180)
        assert res.returncode==0, res.stdout+res.stderr
        _RAN=True
    return out

def load(out, name): return json.loads((out/name).read_text())
def jsonl(out, name): return [json.loads(l) for l in (out/name).read_text().splitlines() if l.strip()]
