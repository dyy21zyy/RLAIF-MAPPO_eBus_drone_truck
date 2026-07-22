from evaluation.scenario_bank import validate_disjoint_banks
from evaluation.paired_evaluation import assert_pairable, PairedScenarioMismatchError, validate_paired_scenarios
import pytest
def test_disjoint_ids_and_artifacts():
    validate_disjoint_banks([{'split':'train','scenarios':[{'scenario_id':'a','artifact_hashes':{'instance.json':'ha'}}]},{'split':'test','scenarios':[{'scenario_id':'b','artifact_hashes':{'instance.json':'hb'}}]}])
    with pytest.raises(ValueError): validate_disjoint_banks([{'split':'train','scenarios':[{'scenario_id':'a','artifact_hashes':{'instance.json':'h'}}]},{'split':'test','scenarios':[{'scenario_id':'a','artifact_hashes':{'instance.json':'x'}}]}])
    with pytest.raises(ValueError): validate_disjoint_banks([{'split':'train','scenarios':[{'scenario_id':'a','artifact_hashes':{'instance.json':'h'}}]},{'split':'test','scenarios':[{'scenario_id':'b','artifact_hashes':{'instance.json':'h'}}]}])
def test_pair_hashes():
    a={'scenario_id':'s','instance_hash':'i','scenario_manifest_hash':'m','artifact_hashes':{'x':'y'},'status':'success','method_id':'a'}
    b=dict(a, method_id='b')
    assert_pairable(a,b); validate_paired_scenarios([a,b])
    with pytest.raises(PairedScenarioMismatchError): assert_pairable(a, dict(b, instance_hash='z'))
    with pytest.raises(PairedScenarioMismatchError): assert_pairable(a, dict(b, scenario_id='t'))
