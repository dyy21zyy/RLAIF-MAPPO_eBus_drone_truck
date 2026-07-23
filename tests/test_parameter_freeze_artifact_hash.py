
from copy import deepcopy
import subprocess, sys
import yaml, pytest
from pathlib import Path
from evaluation.parameter_freeze import *

CFG=load_freeze_template('configs/paper/final_experiment_freeze.template.yaml')

def test_canonical_hash_deterministic(): assert freeze_hash(CFG)==freeze_hash(deepcopy(CFG))
def test_changed_parameter_changes_hash():
    c=deepcopy(CFG); c['training_protocol']['mappo_optimization']['gamma']['value']=0.5
    assert freeze_hash(c)!=freeze_hash(CFG)
def test_existing_artifact_requires_force(tmp_path):
    out=tmp_path/'freeze.json'; out.write_text('{}')
    r=subprocess.run([sys.executable,'-m','experiments.freeze_final_experiment_parameters','--config','configs/paper/final_experiment_freeze.template.yaml','--output',str(out)],capture_output=True,text=True)
    assert r.returncode==2
def test_validate_only_does_not_write(tmp_path):
    out=tmp_path/'freeze.json'
    r=subprocess.run([sys.executable,'-m','experiments.freeze_final_experiment_parameters','--config','configs/paper/final_experiment_freeze.template.yaml','--output',str(out),'--validate-only'],capture_output=True,text=True)
    assert r.returncode==0 and not out.exists()
