import pytest
from rlaif.reward_registry import empty_rlaif_training_totals, update_rlaif_training_totals
from training.reward_contribution import RewardContribution

def test_reconcile_per_agent_totals():
    totals=empty_rlaif_training_totals()
    c=RewardContribution("truck","TRUCK_AVAILABLE",2,5,5,1,.2,.2,2.2,False)
    update_rlaif_training_totals(totals,c)
    assert totals["rlaif_truck_raw"]==5
    assert totals["rlaif_total_weighted"]==pytest.approx(.2)
    assert totals["combined_reward_total"]==pytest.approx(2.2)
