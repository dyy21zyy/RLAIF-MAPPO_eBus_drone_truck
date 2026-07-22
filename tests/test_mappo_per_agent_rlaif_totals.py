from rlaif.reward_registry import empty_rlaif_training_totals, update_rlaif_training_totals
from training.reward_contribution import RewardContribution

def test_per_agent_totals_reconcile_and_disabled_zero():
    t=empty_rlaif_training_totals()
    update_rlaif_training_totals(t, RewardContribution('bus','BUS_STATION_ARRIVAL',2,3,4,1,.5,.5,2.5,False,'h'))
    assert t['rlaif_bus_raw']==3 and t['rlaif_bus_normalized']==4 and t['rlaif_bus_clipped']==1 and t['rlaif_bus_weighted']==.5
    assert t['rlaif_assignment_weighted']==0 and t['combined_reward_total']==2.5
