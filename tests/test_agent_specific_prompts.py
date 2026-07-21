from rlaif.agent_prompt_builders import build_agent_prompt, FACTS

def state(agent,event):
    return {"state_id":"s","agent_type":agent,"event_type":event,"event_time":1,"state_feature_names":["x"],"state_features":[1],"candidate_feature_names":["y"],"candidate_features":[[1],[2]],"candidate_actions":[{"id":0},{"id":1}],"action_masks":[True,True]}

def test_each_prompt_contains_correct_facts():
    events={"assignment":"PARCEL_RELEASE","truck":"TRUCK_AVAILABLE","bus":"BUS_TERMINAL_DEPARTURE","station":"STATION_OPERATION"}
    for agent,event in events.items():
        prompt=build_agent_prompt(state(agent,event),{"original_pair_order":[0,1],"display_order":[0,1]})['prompt_text']
        for fact in FACTS[agent]: assert fact in prompt
    assert 'greener' not in build_agent_prompt(state('assignment','PARCEL_RELEASE'),{"original_pair_order":[0,1]})['prompt_text'].lower()
