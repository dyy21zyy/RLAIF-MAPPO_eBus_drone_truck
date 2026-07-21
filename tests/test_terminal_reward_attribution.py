from envs.reward_ledger import RewardLedger

def test_terminal_residual_uses_decision_chains_not_last_agent_only():
    ledger = RewardLedger()
    ledger.add_cost(event_time=10, component="undelivered", raw_amount=3, weight=2, parcel_ids=["p1","p2"], decision_chain_refs=["assignment:1","truck:2"], source_transition_ids=["terminal"], provenance="terminal_team_distribution")
    entry = ledger.entries[0]
    assert entry.provenance == "terminal_team_distribution"
    assert len(entry.decision_chain_refs) > 1
    assert entry.source_transition_ids != ("station:last",)
