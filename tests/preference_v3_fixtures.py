from training.event_schema import OBSERVATION_SCHEMA_VERSION,CANDIDATE_SCHEMA_VERSION,EVENT_SCHEMA_VERSION

def rec(**kw):
    d={"preference_id":"p1","scenario_id":"s1","episode_id":"e1","state_id":"st1","agent_type":"assignment","event_type":"PARCEL_RELEASE","state_feature_names":["x","y"],"state_features":[1.0,2.0],"candidate_a_feature_names":["c","d"],"candidate_a_features":[3.0,4.0],"candidate_b_feature_names":["c","d"],"candidate_b_features":[1.0,2.0],"original_candidate_a_id":"a","original_candidate_b_id":"b","displayed_first_candidate_id":"a","displayed_second_candidate_id":"b","original_outcome":"candidate_a","label_source":"external_evaluator_api","evaluator_model":"m","evaluator_prompt_version":"v","observation_schema_version":OBSERVATION_SCHEMA_VERSION,"candidate_schema_version":CANDIDATE_SCHEMA_VERSION,"event_schema_version":EVENT_SCHEMA_VERSION,"created_at":"2026-07-22T00:00:00Z"}
    d.update(kw); return d
