from rlaif.preference_schema_v2 import preference_state_from_observation, validate_preference_state, RLAIF_AGENT_TYPES

def obs(agent,event):
    return {"agent_id":agent,"event_type":event,"time_min":0,"feature_names":["f"],"features":[0],"candidate_feature_names":["c"],"candidate_features":[[0],[1]],"candidate_actions":[{"a":0},{"a":1}],"action_mask":[True,True]}

def test_all_four_agents_create_preference_state_records():
    events={"assignment":"PARCEL_RELEASE","truck":"TRUCK_AVAILABLE","bus":"BUS_TERMINAL_DEPARTURE","station":"STATION_OPERATION"}
    got=set()
    for i,(a,e) in enumerate(events.items()):
        r=preference_state_from_observation(obs(a,e),scenario_id='sc',episode_id=1,decision_id=i,collection_policy={"source":"random"})
        validate_preference_state(r); got.add(r['agent_type'])
    assert got==RLAIF_AGENT_TYPES

def test_bus_loading_and_charging_preserve_event_metadata():
    for e in ["BUS_TERMINAL_DEPARTURE","BUS_STATION_ARRIVAL"]:
        assert preference_state_from_observation(obs('bus',e),scenario_id='s',episode_id=1,decision_id=e)['event_type']==e
