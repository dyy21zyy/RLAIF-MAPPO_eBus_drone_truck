import pytest
from evaluation.config_difference_validation import validate_sensitivity_config_difference, validate_ablation_override_applied, UnexpectedSensitivityConfigDifferenceError

def test_declared_sensitivity_path_accepted(): validate_sensitivity_config_difference({'a':{'b':1},'x':2},{'a':{'b':3},'x':2},'a.b')
def test_unrelated_behavioral_difference_fails():
    with pytest.raises(UnexpectedSensitivityConfigDifferenceError): validate_sensitivity_config_difference({'a':1,'x':2},{'a':1,'x':3},'a')
def test_output_path_difference_ignored(): validate_sensitivity_config_difference({'output':{'path':'a'}},{'output':{'path':'b'}},'x')
def test_timestamp_difference_ignored(): validate_sensitivity_config_difference({'created_at':'a'},{'created_at':'b'},'x')
def test_ablation_override_must_change_config():
    with pytest.raises(UnexpectedSensitivityConfigDifferenceError): validate_ablation_override_applied({'a':1},{'a':1},['a'])
