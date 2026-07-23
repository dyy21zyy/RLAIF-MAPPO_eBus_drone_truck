
from copy import deepcopy
import subprocess, sys
import yaml, pytest
from pathlib import Path
from evaluation.parameter_freeze import *

CFG=load_freeze_template('configs/paper/final_experiment_freeze.template.yaml')

def test_every_canonical_reward_component_exists(): assert set(CFG['reward_weights']['components'])==set(CANONICAL_REWARD_COMPONENTS)
def test_unknown_reward_component_fails():
    c=deepcopy(CFG['reward_weights']); c['components']['bad']={'weight':1,'physical_unit':'u','sign_convention':'penalty_nonnegative','source':'s','rationale':'r'}
    with pytest.raises(ParameterFreezeError): validate_reward_weights(c)
def test_missing_reward_component_fails():
    c=deepcopy(CFG['reward_weights']); c['components'].pop('energy_cost')
    with pytest.raises(ParameterFreezeError): validate_reward_weights(c)
def test_all_zero_reward_weights_fail():
    c=deepcopy(CFG['reward_weights'])
    for v in c['components'].values(): v['weight']=0
    with pytest.raises(ParameterFreezeError): validate_reward_weights(c)
def test_scale_artifact_matches_train_bank_placeholder_lineage(): assert CFG['reward_reference_scale']['training_bank_hash']['value']==CFG['scenario_protocol']['train_bank_hash']['value']
