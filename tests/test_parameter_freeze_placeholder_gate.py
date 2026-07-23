
from copy import deepcopy
import subprocess, sys
import yaml, pytest
from pathlib import Path
from evaluation.parameter_freeze import *

CFG=load_freeze_template('configs/paper/final_experiment_freeze.template.yaml')

def test_placeholder_blocks_ready_status(): assert validate_freeze_template(CFG)['status']=='PARAMETER_FREEZE_TEMPLATE_VALID'
def test_placeholder_classification(): assert all(p['resolved_status']=='unresolved_placeholder' for p in unresolved_placeholders(CFG))
def test_required_blocked_statuses_present():
    b=set(validate_freeze_template(CFG)['blocked_statuses']); assert {'BLOCKED_SCENARIO_BANK_HASH','BLOCKED_REWARD_SCALE_HASH','BLOCKED_REWARD_CHECKPOINT_HASH'} <= b
def test_readiness_mode_raises():
    with pytest.raises(UnresolvedParameterFreezeError): validate_freeze_template(CFG, readiness=True)
def test_template_can_pass_structural_validation(): assert validate_freeze_template(CFG)['parameter_count'] > 25
