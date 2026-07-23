
from copy import deepcopy
import subprocess, sys
import yaml, pytest
from pathlib import Path
from evaluation.parameter_freeze import *

CFG=load_freeze_template('configs/paper/final_experiment_freeze.template.yaml')

def test_expected_row_count_is_derived(): assert expected_benchmark_rows(CFG['evaluation_protocol'],[1,2,3])['total']==1400
def test_learned_methods_use_three_seeds(): assert all(m['seeds']==[1,2,3] for m in CFG['evaluation_protocol']['method_matrix']['learned'])
def test_heuristics_do_not_require_checkpoints(): assert all(not m['requires_checkpoint'] for m in CFG['evaluation_protocol']['method_matrix']['heuristics'])
def test_deterministic_evaluation_frozen(): assert CFG['evaluation_protocol']['deterministic_policy_evaluation']['value'] is True
def test_paired_evaluation_required(): assert CFG['evaluation_protocol']['paired_evaluation']['value'] is True
