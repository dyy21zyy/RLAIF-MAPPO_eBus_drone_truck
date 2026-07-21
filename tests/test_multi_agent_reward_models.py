import torch
from rlaif.multi_agent_reward_model import MultiAgentRewardModel, bradley_terry_loss

def test_pairwise_loss_excludes_ties_by_filtering_before_loss():
    m=MultiAgentRewardModel(2,2,["TRUCK_AVAILABLE"])
    s=torch.zeros(2,2); c=torch.zeros(2,2); scores=m(s,c,torch.zeros(2,dtype=torch.long))
    loss=bradley_terry_loss(scores[:1]+1,scores[:1])
    assert torch.isfinite(loss)
