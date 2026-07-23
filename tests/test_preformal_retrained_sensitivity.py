import pytest
from evaluation.preformal_part3_gates import BenchmarkIntegrityError, validate_sensitivity_protocol

def test_retrained_policy_hashes_separate_and_protocol_not_mixed():
    validate_sensitivity_protocol([{'protocol':'retrained_policy_sensitivity','factor_value':1,'policy_checkpoint_hash':'h1'},{'protocol':'retrained_policy_sensitivity','factor_value':2,'policy_checkpoint_hash':'h2'}],'retrained_policy_sensitivity')
    with pytest.raises(BenchmarkIntegrityError): validate_sensitivity_protocol([{'protocol':'fixed_policy_robustness','factor_value':1,'policy_checkpoint_hash':'h1'},{'protocol':'retrained_policy_sensitivity','factor_value':2,'policy_checkpoint_hash':'h2'}],'retrained_policy_sensitivity')
