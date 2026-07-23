import pytest
from evaluation.preformal_part3_gates import BenchmarkIntegrityError, validate_sensitivity_protocol

def test_fixed_policy_hash_identical():
    validate_sensitivity_protocol([{'protocol':'fixed_policy_robustness','factor_value':1,'policy_checkpoint_hash':'h'},{'protocol':'fixed_policy_robustness','factor_value':2,'policy_checkpoint_hash':'h'}],'fixed_policy_robustness')
    with pytest.raises(BenchmarkIntegrityError): validate_sensitivity_protocol([{'protocol':'fixed_policy_robustness','factor_value':1,'policy_checkpoint_hash':'h1'},{'protocol':'fixed_policy_robustness','factor_value':2,'policy_checkpoint_hash':'h2'}],'fixed_policy_robustness')
