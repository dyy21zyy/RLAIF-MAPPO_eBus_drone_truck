import pytest
from evaluation.preformal_part3_gates import BenchmarkIntegrityError, validate_checkpoint_separation

def test_retraining_required_checkpoints_differ():
    validate_checkpoint_separation([{'variant_id':'env','retraining_required':True,'resolved_training_config_hash':'a','checkpoint_path':'a','checkpoint_hash':'h1','training_log':'l','evaluation_dir':'e'},{'variant_id':'ra','retraining_required':True,'resolved_training_config_hash':'b','checkpoint_path':'b','checkpoint_hash':'h2','training_log':'l','evaluation_dir':'e'}])
    with pytest.raises(BenchmarkIntegrityError): validate_checkpoint_separation([{'variant_id':'env','retraining_required':True,'resolved_training_config_hash':'a','checkpoint_path':'a','checkpoint_hash':'h','training_log':'l','evaluation_dir':'e'},{'variant_id':'ra','retraining_required':True,'resolved_training_config_hash':'b','checkpoint_path':'b','checkpoint_hash':'h','training_log':'l','evaluation_dir':'e'}])
