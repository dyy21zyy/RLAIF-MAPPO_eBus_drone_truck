import math, torch
from rlaif.multi_agent_reward_model import bradley_terry_loss
def test_equal_scores_log2(): assert torch.isclose(bradley_terry_loss(torch.tensor([1.]),torch.tensor([1.]),torch.tensor([1.])), torch.tensor(math.log(2.)), atol=1e-6)
def test_deterministic_loss_and_reversed_larger():
 l=bradley_terry_loss(torch.tensor([2.]),torch.tensor([.5]),torch.tensor([1.])); exp=-torch.log(torch.sigmoid(torch.tensor(1.5))); assert torch.isclose(l,exp); assert bradley_terry_loss(torch.tensor([.5]),torch.tensor([2.]),torch.tensor([1.]))>l
def test_gradients_reach_both_paths():
 a=torch.tensor([0.],requires_grad=True); b=torch.tensor([0.],requires_grad=True); bradley_terry_loss(a,b,torch.tensor([1.])).backward(); assert a.grad is not None and b.grad is not None and a.grad.item()!=0 and b.grad.item()!=0
