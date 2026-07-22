import pytest, torch
from training.mappo_networks import CandidateScoringActor

def test_forward_requires_event_ids_and_masks():
    actor=CandidateScoringActor(2,2,event_embedding_dim=4)
    with pytest.raises(TypeError): actor(torch.zeros(1,2), torch.zeros(1,2,2))
    logits=actor(torch.zeros(1,2), torch.tensor([0]), torch.zeros(1,2,2), torch.tensor([[True,False]]))
    assert logits[0,1] < -1e8
