from envs.reward_ledger import RewardLedger

def test_reward_reference_scale_applies_to_weighted_amount():
    l=RewardLedger(); r=l.add_cost(event_time=0, component='truck_cost', raw_amount=10, weight=2, reference_scale=5)
    e=l.entries[0]
    assert e.normalized_amount == 2
    assert e.weighted_amount == 4
    assert r == -4 and l.reward_sum() == -4
