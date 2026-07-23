
from copy import deepcopy
import subprocess, sys
import yaml, pytest
from pathlib import Path
from evaluation.parameter_freeze import *

CFG=load_freeze_template('configs/paper/final_experiment_freeze.template.yaml')

def test_test_count_100(): assert CFG['scenario_protocol']['test_scenario_count']['value']==100
def test_training_seeds(): assert CFG['seed_protocol']['training_seeds']['value']==[1,2,3]
def test_seed_namespaces_distinct():
    vals=[v['value'] for v in CFG['seed_protocol']['namespaces'].values()]
    assert len(vals)==len(set(vals))
def test_unresolved_bank_hash_blocks_readiness():
    r=validate_freeze_template(CFG); assert 'BLOCKED_SCENARIO_BANK_HASH' in r['blocked_statuses']
def test_test_bank_not_in_training(): assert CFG['scenario_protocol']['training_uses_test_bank']['value'] is False
