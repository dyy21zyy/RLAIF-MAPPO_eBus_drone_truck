
from copy import deepcopy
import subprocess, sys
import yaml, pytest
from pathlib import Path
from evaluation.parameter_freeze import *

CFG=load_freeze_template('configs/paper/final_experiment_freeze.template.yaml')

def test_canonical_agents_are_frozen(): assert tuple(CFG['scientific_contract']['agents']['value'])==CANONICAL_AGENTS
def test_canonical_decision_events_are_frozen(): assert CFG['scientific_contract']['decision_events']['value']==DECISION_EVENT_TO_AGENT
def test_station_baseline_dispatch_idle(): assert tuple(CFG['scientific_contract']['station_baseline_actions']['value'])==STATION_BASELINE_ACTIONS
def test_assignment_only_primary_baseline(): assert CFG['paper_method_contract']['primary_rlaif_method']['value']=='mappo_rlaif_assignment'
def test_reward_models_are_not_action_selectors(): assert CFG['paper_method_contract']['reward_model_role']['value']=='score_selected_transition_only'
