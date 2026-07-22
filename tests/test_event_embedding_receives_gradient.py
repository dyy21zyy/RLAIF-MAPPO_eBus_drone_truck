import torch
from training.mappo_networks import CandidateScoringActor

def test_event_embedding_receives_gradient_and_updates():
    actor=CandidateScoringActor(2,2,hidden_dims=[4],event_embedding_dim=2)
    opt=torch.optim.SGD(actor.parameters(), lr=0.1)
    before=actor.event_embedding.weight.detach().clone()
    lp, ent=actor.evaluate_actions(torch.zeros(2,2), torch.tensor([2,3]), torch.tensor([[[1.,0.],[0.,1.]],[[0.,1.],[1.,0.]]]), torch.ones(2,2,dtype=torch.bool), torch.tensor([0,1]))
    loss=-(lp.mean()+0.01*ent.mean()); opt.zero_grad(); loss.backward()
    grad=actor.event_embedding.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad[[2,3]].abs().sum()>0
    opt.step()
    assert (actor.event_embedding.weight.detach()-before).abs().sum()>0
