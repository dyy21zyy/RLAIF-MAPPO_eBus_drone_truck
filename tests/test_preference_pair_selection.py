from rlaif.pair_selector import select_informative_pairs

def test_pair_selection_not_fixed_and_uses_signals():
    st={"state_id":"s","agent_type":"truck","event_type":"TRUCK_AVAILABLE","candidate_feature_names":["estimated_lateness","travel_time"],"candidate_features":[[1,2],[1.1,2],[9,9]],"action_masks":[True,True,True]}
    pairs=select_informative_pairs(st,max_pairs=2)
    assert pairs[0]['original_pair_order']==[0,1]
    assert 'entropy' in pairs[0]['selection_signals']
