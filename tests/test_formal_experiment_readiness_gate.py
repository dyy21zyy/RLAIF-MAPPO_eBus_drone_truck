from experiments.validate_formal_experiment_readiness import validate_readiness, CONFIG_VALID_ARTIFACTS_MISSING, CONFIG_INVALID
import yaml
def test_missing_bank(tmp_path):
    c=tmp_path/'c.yaml'; c.write_text(yaml.safe_dump({'run_classification':'formal','fallback':False,'paired_evaluation':True,'scenario_bank':{'manifest':str(tmp_path/'missing.json'),'expected_count':1},'methods':[{'method_id':'truck_direct_heuristic'}]}))
    assert validate_readiness(c)['status']==CONFIG_VALID_ARTIFACTS_MISSING
def test_fallback_blocks(tmp_path):
    c=tmp_path/'c.yaml'; c.write_text(yaml.safe_dump({'run_classification':'formal','fallback':True,'scenario_bank':{'manifest':str(tmp_path/'missing.json')},'methods':[]}))
    assert validate_readiness(c)['status']==CONFIG_INVALID
