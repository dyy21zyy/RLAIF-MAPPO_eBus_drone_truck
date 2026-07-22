import pytest
from experiments.run_paper_sensitivity import validate_sensitivity, sensitivity_row
def test_modes():
    validate_sensitivity({'experiments':[{'mode':'fixed_policy_robustness','policy_checkpoint_hash':'h','dimensions':[]}]})
    r=sensitivity_row('fixed_policy_robustness','p',1,'h','s'); assert r['sensitivity_mode']
    with pytest.raises(ValueError): validate_sensitivity({'experiments':[{'mode':'fixed_policy_robustness','policy_checkpoint_hash':'h','combined_table':True},{'mode':'retrained_policy_sensitivity','dimensions':[{'name':'x'}]}]})
