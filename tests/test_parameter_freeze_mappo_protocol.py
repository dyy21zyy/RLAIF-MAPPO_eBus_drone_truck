
from copy import deepcopy
import subprocess, sys
import yaml, pytest
from pathlib import Path
from evaluation.parameter_freeze import *

CFG=load_freeze_template('configs/paper/final_experiment_freeze.template.yaml')

def test_mappo_methods_share_network_architecture(): assert CFG['network_architecture']['methods']==['mappo_env','mappo_rlaif_assignment','mappo_rlaif_all']
def test_mappo_optimizer_frozen(): assert CFG['training_protocol']['mappo_optimization']['lr_actor']['value']==0.00005
def test_total_episodes_3000(): assert CFG['training_protocol']['mappo_optimization']['total_training_episodes']['value']==3000
def test_event_time_discount_frozen():
    e=CFG['training_protocol']['event_time_discount']; assert e['enabled']['value'] is True and 'gamma^' in 'gamma_i = gamma^(delta_t/tau)'
def test_method_specific_unauthorized_difference_fails():
    from evaluation.method_config_difference import validate_method_config_differences, UnexpectedMethodConfigDifferenceError
    c=yaml.safe_load(Path('configs/paper/method_difference_contract.yaml').read_text())
    a={'training':{'lr_actor':1},'rlaif':{'enabled':False}}; b={'training':{'lr_actor':2},'rlaif':{'enabled':True}}
    with pytest.raises(UnexpectedMethodConfigDifferenceError): validate_method_config_differences(a,b,baseline_method='mappo_env',comparison_method='mappo_rlaif_assignment',contract=c)
