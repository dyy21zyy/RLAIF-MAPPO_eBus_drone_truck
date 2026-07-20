"""Stage 7 masked actor and centralized critic tests."""
import pytest
torch=pytest.importorskip('torch')
from training.mappo_networks import (
    AssignmentActor,
    BusActor,
    CandidateScoringActor,
    CentralizedCritic,
    build_actor_registry,
)

def test_assignment_actor_respects_mask():
    actor=AssignmentActor(3,2,[8]); mask=[False,True,False,False,False]
    assert all(actor.act([0,0,0],mask)[0]==1 for _ in range(20))

def test_bus_actor_respects_mask():
    actor=BusActor(2,[8]); mask=[False]*9; mask[7]=True
    assert all(actor.act([0,0],mask)[0]==7 for _ in range(20))

def test_centralized_critic_returns_scalar_per_state():
    critic=CentralizedCritic(4,[8]); value=critic(torch.zeros(4))
    assert value.shape==(1,) and torch.isfinite(value).all()

def test_candidate_scoring_actor_respects_variable_candidate_mask():
    actor = CandidateScoringActor(obs_dim=3, candidate_feature_dim=2, hidden_dims=[8])
    observation = torch.zeros(3)
    candidates = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    mask = torch.tensor([False, True, False])
    assert all(actor.act(observation, candidates, mask)[0] == 1 for _ in range(20))

def test_actor_registry_contains_four_agent_types():
    registry = build_actor_registry(
        {
            "assignment": (4, 3),
            "truck": (5, 3),
            "bus": (6, 3),
            "station": (7, 3),
        },
        {"default": [8]},
    )
    assert set(registry) == {"assignment", "truck", "bus", "station"}
