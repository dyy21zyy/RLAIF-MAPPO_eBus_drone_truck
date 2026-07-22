import pytest
from experiments.run_ablation_matrix import validate_matrix, MatrixValidationError
from experiments.run_sensitivity_matrix import validate_matrix as validate_s, SensitivityMatrixValidationError

def base():
    return {'experiment_kind':'ablation','baseline_variant_id':'b','scenario_banks':{},'training_seeds':[1],'variants':[{'variant_id':'b','execution_mode':'retrain_and_evaluate','training_config':'x','benchmark_method_id':'mappo_env'}]}
def test_duplicate_variant_ids_fail():
    c=base(); c['variants'].append(dict(c['variants'][0]))
    with pytest.raises(MatrixValidationError): validate_matrix(c)
def test_unknown_execution_mode_fails():
    c=base(); c['variants'][0]['execution_mode']='bad'
    with pytest.raises(MatrixValidationError): validate_matrix(c)
def test_missing_baseline_fails():
    c=base(); c.pop('baseline_variant_id'); c['variants'][0]['variant_id']='x'
    with pytest.raises(MatrixValidationError): validate_matrix(c)
def test_retraining_variant_requires_training_config():
    c=base(); c['variants'][0].pop('training_config')
    with pytest.raises(MatrixValidationError): validate_matrix(c)
def test_fixed_policy_variant_requires_checkpoint():
    c=base(); c['variants'][0]['execution_mode']='fixed_policy_evaluate'
    with pytest.raises(MatrixValidationError): validate_matrix(c)
def test_identical_ablation_config_fails():
    c=base(); c['variants'][0]['declared_override_paths']=['a.b']
    with pytest.raises(MatrixValidationError): validate_matrix(c)
def test_unknown_sensitivity_protocol_fails():
    c={'experiment_kind':'sensitivity','protocols':[{'protocol_id':'bad','execution_mode':'fixed_policy_evaluate'}],'factors':[]}
    with pytest.raises(SensitivityMatrixValidationError): validate_s(c)
