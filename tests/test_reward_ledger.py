from envs.reward_ledger import RewardLedger, REWARD_COMPONENTS

def test_reward_ledger_sums_to_environment_reward():
    ledger = RewardLedger()
    r1 = ledger.add_cost(event_time=1, component="passenger_delay", raw_amount=2, weight=3, source_transition_ids=["a:1"])
    r2 = ledger.add_cost(event_time=2, component="undelivered", raw_amount=1, weight=5, parcel_ids=["p1"], decision_chain_refs=["assignment:1"], provenance="terminal_team_distribution")
    assert r1 + r2 == ledger.reward_sum() == -11
    assert {e.component for e in ledger.entries} <= REWARD_COMPONENTS
