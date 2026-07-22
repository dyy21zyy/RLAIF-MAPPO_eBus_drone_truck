from rlaif.reward_registry import empty_rlaif_training_totals, update_rlaif_training_totals
from training.reward_contribution import RewardContribution

def test_only_active_agent_updates():
    totals=empty_rlaif_training_totals(); update_rlaif_training_totals(totals, RewardContribution("bus","BUS_STATION_ARRIVAL",0,1,1,1,1,1,1,False))
    assert totals["rlaif_bus_weighted"]==1 and totals["rlaif_assignment_weighted"]==0
