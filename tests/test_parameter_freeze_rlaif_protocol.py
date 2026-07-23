
from copy import deepcopy
import subprocess, sys
import yaml, pytest
from pathlib import Path
from evaluation.parameter_freeze import *

CFG=load_freeze_template('configs/paper/final_experiment_freeze.template.yaml')

def test_assignment_baseline_enables_assignment_only():
    a=CFG['rlaif_parameters']['methods']['mappo_rlaif_assignment']['agents']; assert [k for k,v in a.items() if v['enabled']]==['assignment']
def test_full_scope_enables_all_four():
    a=CFG['rlaif_parameters']['methods']['mappo_rlaif_all']['agents']; assert set(k for k,v in a.items() if v['enabled'])==set(CANONICAL_AGENTS)
def test_fallback_false(): assert CFG['rlaif_parameters']['fallback_to_env_reward']['value'] is False
def test_normalization_order_frozen(): assert CFG['rlaif_parameters']['normalization_order']['value'][0]=='raw_reward_model_score'
def test_learned_reward_not_env_scaled(): assert CFG['rlaif_parameters']['learned_reward_uses_environment_reference_scale']['value'] is False
