import torch
from training.mappo_networks import CandidateScoringActor
from training.event_schema import decision_event_id

def test_same_inputs_different_bus_event_changes_logits():
    actor=CandidateScoringActor(2,2,hidden_dims=[],event_embedding_dim=2)
    with torch.no_grad():
        actor.event_embedding.weight.zero_()
        actor.event_embedding.weight[decision_event_id("BUS_TERMINAL_DEPARTURE")]=torch.tensor([1.,0.])
        actor.event_embedding.weight[decision_event_id("BUS_STATION_ARRIVAL")]=torch.tensor([0.,1.])
        lin=actor.scorer[0]; lin.weight.zero_(); lin.bias.zero_(); lin.weight[0,2]=1.; lin.weight[0,3]=2.
    obs=torch.tensor([[0.,0.]])
    cand=torch.tensor([[[0.,0.],[0.,0.]]])
    mask=torch.tensor([[True,True]])
    a=actor(obs, torch.tensor([2]), cand, mask)
    b=actor(obs, torch.tensor([3]), cand, mask)
    assert not torch.allclose(a,b)
