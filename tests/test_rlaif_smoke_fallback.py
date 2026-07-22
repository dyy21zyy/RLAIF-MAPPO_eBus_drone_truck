from rlaif.reward_registry import RewardRegistry

def test_smoke_fallback_zeroes_learned(caplog):
    r=RewardRegistry({"rlaif":{"enabled":True,"fallback_to_env_reward":True,"fail_on_invalid_reward_model":False,"agents":{"assignment":{"enabled":True,"checkpoint":"missing.pt","lambda":.2,"reward_clip":2}}}})
    c=r.score_transition(agent_type="assignment",event_type="PARCEL_RELEASE",environment_reward=3,state_features=[0],candidate_features=[0],selected_action_index=0,formal_mode=False)
    assert c.used_fallback and c.total_reward==3 and c.weighted_learned_contribution==0
