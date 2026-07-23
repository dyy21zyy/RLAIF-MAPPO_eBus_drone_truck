import pytest
from evaluation.preformal_part3_gates import PreformalPairedScenarioMismatchError, assert_preformal_pairable

def r(**kw):
    d={'scenario_id':'s','scenario_content_hash':'c','instance_hash':'i','scenario_manifest_hash':'m','scenario_bank_hash':'b','artifact_hashes':{'parcel_hash':'p','passenger_demand_hash':'q'}}; d.update(kw); return d

def test_strict_artifact_pairing():
    assert_preformal_pairable(r(),r())
    with pytest.raises(PreformalPairedScenarioMismatchError): assert_preformal_pairable(r(),r(instance_hash='x'))
    with pytest.raises(PreformalPairedScenarioMismatchError): assert_preformal_pairable(r(),r(artifact_hashes={'parcel_hash':'x','passenger_demand_hash':'q'}))
