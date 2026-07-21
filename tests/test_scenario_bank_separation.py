import json
from evaluation.scenario_bank import validate_disjoint_banks, write_bank_manifest

def test_scenario_banks_are_disjoint_and_data_differs(tmp_path):
    a={"bank":"train","scenarios":[{"scenario_id":"train_1","seed_tuple":{"a":1},"artifact_hashes":{"instance.json":"h1"}}]}
    b={"bank":"test","scenarios":[{"scenario_id":"test_1","seed_tuple":{"a":2},"artifact_hashes":{"instance.json":"h2"}}]}
    validate_disjoint_banks([a,b])
    assert a['scenarios'][0]['artifact_hashes']['instance.json'] != b['scenarios'][0]['artifact_hashes']['instance.json']

def test_duplicate_scenario_ids_rejected():
    m1={"bank":"train","scenarios":[{"scenario_id":"x","seed_tuple":{"a":1}}]}
    m2={"bank":"test","scenarios":[{"scenario_id":"x","seed_tuple":{"a":2}}]}
    import pytest
    with pytest.raises(ValueError): validate_disjoint_banks([m1,m2])
