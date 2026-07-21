import pytest
pytest.importorskip("torch")
import torch
from training.mappo_networks import CandidateScoringActor
from training.mappo_trainer import _padded_candidate_batch
from training.mappo_buffer import AsyncTransition

def _t(n):
    return AsyncTransition("truck", [0.0], [0.0], 0, [True]+[False]*(n-1), [[float(i), 1.0] for i in range(n)], ("id","x"), 0,0,0,False,[0.0],"TRUCK_AVAILABLE",0)

def test_variable_candidate_sets_pad_and_mask_zero_probability():
    cands, masks = _padded_candidate_batch([_t(1), _t(3)])
    assert cands.shape == (2,3,2)
    actor = CandidateScoringActor(1,2,[4])
    dist = actor.distribution(torch.zeros(2,1), cands, masks)
    assert torch.all(dist.probs[~masks] == 0)
