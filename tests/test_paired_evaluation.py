import pytest
from evaluation.paired_evaluation import validate_paired_scenarios
from evaluation.statistics import paired_difference

def test_same_paired_scenario_used_by_all_methods():
    rows=[{"method_name":"a","scenario_id":"s1","status":"success"},{"method_name":"b","scenario_id":"s1","status":"success"}]
    assert validate_paired_scenarios(rows)

def test_paired_differences_require_matched_ids():
    with pytest.raises(ValueError): paired_difference([{"scenario_id":"s1","status":"success","x":1}],[{"scenario_id":"s2","status":"success","x":2}],"x")
