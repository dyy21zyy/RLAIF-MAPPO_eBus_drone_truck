
from copy import deepcopy
import subprocess, sys
import yaml, pytest
from pathlib import Path
from evaluation.parameter_freeze import *

CFG=load_freeze_template('configs/paper/final_experiment_freeze.template.yaml')

from evaluation.method_config_difference import validate_method_config_differences, UnexpectedMethodConfigDifferenceError
CONTRACT=yaml.safe_load(Path('configs/paper/method_difference_contract.yaml').read_text())
def base(): return {'mode':'environment_reward','training':{'lr_actor':5e-5,'gamma':.997,'total_episodes':3000},'networks':{'h':[256]},'scenario_banks':{'train':'A'},'reward':{'w':1},'rlaif':{'enabled':False},'output':{'root':'a'}}
def rlaif():
    x=base(); x['mode']='rlaif'; x['rlaif']={'enabled':True,'scope':'assignment','agents':{'assignment':{'lambda':1}}}; x['output']={'root':'b'}; return x
def test_allowed_rlaif_difference_passes(): assert validate_method_config_differences(base(),rlaif(),baseline_method='mappo_env',comparison_method='mappo_rlaif_assignment',contract=CONTRACT)['comparison_status']=='passed'
def test_changed_learning_rate_fails():
    b=rlaif(); b['training']['lr_actor']=1e-4
    with pytest.raises(UnexpectedMethodConfigDifferenceError): validate_method_config_differences(base(),b,baseline_method='mappo_env',comparison_method='mappo_rlaif_assignment',contract=CONTRACT)
def test_changed_network_size_fails():
    b=rlaif(); b['networks']['h']=[512]
    with pytest.raises(UnexpectedMethodConfigDifferenceError): validate_method_config_differences(base(),b,baseline_method='mappo_env',comparison_method='mappo_rlaif_assignment',contract=CONTRACT)
def test_changed_train_bank_fails():
    b=rlaif(); b['scenario_banks']['train']='B'
    with pytest.raises(UnexpectedMethodConfigDifferenceError): validate_method_config_differences(base(),b,baseline_method='mappo_env',comparison_method='mappo_rlaif_assignment',contract=CONTRACT)
def test_identical_env_and_rlaif_configs_fail():
    with pytest.raises(UnexpectedMethodConfigDifferenceError): validate_method_config_differences(base(),base(),baseline_method='mappo_env',comparison_method='mappo_rlaif_assignment',contract=CONTRACT)
