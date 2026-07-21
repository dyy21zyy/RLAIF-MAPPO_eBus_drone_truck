import pytest
from experiments.run_paper_sensitivity import validate_sensitivity

def test_fixed_and_retrained_sensitivity_are_separate():
    cfg={"experiments":[{"mode":"fixed_policy_robustness","combined_table":True,"dimensions":[{"name":"parcel_count"}]},{"mode":"retrained_policy_sensitivity","combined_table":True,"dimensions":[{"name":"truck_count"}]}]}
    with pytest.raises(ValueError): validate_sensitivity(cfg)
