"""Stage 7 masked actor and centralized critic tests."""
import pytest
torch=pytest.importorskip('torch')
from training.mappo_networks import AssignmentActor, BusActor, CentralizedCritic

def test_assignment_actor_respects_mask():
    actor=AssignmentActor(3,2,[8]); mask=[False,True,False,False,False]
    assert all(actor.act([0,0,0],mask)[0]==1 for _ in range(20))

def test_bus_actor_respects_mask():
    actor=BusActor(2,[8]); mask=[False]*9; mask[7]=True
    assert all(actor.act([0,0],mask)[0]==7 for _ in range(20))

def test_centralized_critic_returns_scalar_per_state():
    critic=CentralizedCritic(4,[8]); value=critic(torch.zeros(4))
    assert value.shape==(1,) and torch.isfinite(value).all()
