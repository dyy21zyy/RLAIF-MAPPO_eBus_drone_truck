"""Experiment artifact manifest with provenance hashes."""
from __future__ import annotations
import hashlib,json,platform,subprocess,time
from pathlib import Path
from typing import Any
REQUIRED=("git_commit","dirty_status","resolved_config","scenario_manifest_hash","preference_data_hash","reward_checkpoint_hashes","policy_checkpoint_hash","training_seed","evaluation_seeds","python_version","pytorch_version","hardware","start_time","end_time","runtime_seconds","status","failure_reason")

def sha256_file(path):
    p=Path(path)
    if not p.is_file(): return None
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def git_commit():
    try: return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except Exception: return None

def dirty_status():
    try: return subprocess.check_output(['git','status','--short'],text=True).strip()
    except Exception: return 'unknown'

def torch_version():
    try:
        import torch; return torch.__version__
    except Exception: return None

def build_manifest(**kwargs:Any)->dict[str,Any]:
    now=time.time(); start=kwargs.get('start_time') or now; end=kwargs.get('end_time') or now
    m={"git_commit":git_commit(),"dirty_status":dirty_status(),"resolved_config":{},"scenario_manifest_hash":None,"preference_data_hash":None,"reward_checkpoint_hashes":{},"policy_checkpoint_hash":None,"training_seed":None,"evaluation_seeds":[],"python_version":platform.python_version(),"pytorch_version":torch_version(),"hardware":platform.platform(),"start_time":start,"end_time":end,"runtime_seconds":float(end)-float(start),"status":"success","failure_reason":""}
    m.update(kwargs); missing=[k for k in REQUIRED if k not in m]
    if missing: raise ValueError(f"manifest missing fields: {missing}")
    return m

def write_manifest(path:Path, **kwargs):
    m=build_manifest(**kwargs); Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(m,indent=2,sort_keys=True)+'\n'); return m
