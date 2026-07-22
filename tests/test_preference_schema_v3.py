import math, pytest
from rlaif.preference_schema_v3 import parse_preference_record
from tests.preference_v3_fixtures import rec

def test_valid_record_loads(): assert parse_preference_record(rec()).preference_id=='p1'
def test_unsupported_agent_fails():
    with pytest.raises(ValueError): parse_preference_record(rec(agent_type='drone'))
def test_unsupported_event_fails():
    with pytest.raises(ValueError): parse_preference_record(rec(event_type='NOPE'))
def test_agent_event_mismatch_fails():
    with pytest.raises(ValueError): parse_preference_record(rec(agent_type='truck'))
def test_nonfinite_features_fail():
    with pytest.raises(ValueError): parse_preference_record(rec(state_features=[math.inf,0]))
def test_feature_name_vector_length_mismatch_fails():
    with pytest.raises(ValueError): parse_preference_record(rec(state_features=[1]))
def test_candidate_a_and_b_cannot_be_identical():
    with pytest.raises(ValueError): parse_preference_record(rec(original_candidate_b_id='a', displayed_second_candidate_id='a'))
def test_wrong_schema_version_fails():
    with pytest.raises(ValueError): parse_preference_record(rec(observation_schema_version=999))
