
from copy import deepcopy
import subprocess, sys
import yaml, pytest
from pathlib import Path
from evaluation.parameter_freeze import *

CFG=load_freeze_template('configs/paper/final_experiment_freeze.template.yaml')

def test_template_structural_validation_passes():
    assert validate_freeze_template(CFG)['status']=='PARAMETER_FREEZE_TEMPLATE_VALID'
def test_every_parameter_has_category_source_rationale():
    params=extract_frozen_parameters(CFG); assert params
    validate_frozen_parameters(params)
def test_unknown_category_fails():
    with pytest.raises(ParameterFreezeError): validate_frozen_parameters([FrozenParameter('x',1,'bad','s','r')])
def test_duplicate_key_fails():
    p=FrozenParameter('x',1,'scientific_fixed','s','r')
    with pytest.raises(DuplicateFrozenParameterError): validate_frozen_parameters([p,p])
