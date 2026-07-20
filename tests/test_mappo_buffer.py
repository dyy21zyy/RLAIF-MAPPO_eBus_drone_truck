"""Asynchronous MAPPO buffer tests."""
import numpy as np
import pytest
from training.mappo_buffer import AsyncMAPPOBuffer, AsyncTransition

def transition(agent='assignment', event='PARCEL_RELEASE', episode=0):
    return AsyncTransition(
        agent,
        [0.0],
        [0.0],
        0,
        [True],
        [[1.0, 0.0]],
        ("productive", "idle"),
        -0.2,
        0.1,
        1.0,
        False,
        [0.1],
        event,
        2.0,
        episode_id=episode,
    )

def test_one_active_agent_transition_per_event():
    buffer=AsyncMAPPOBuffer(); buffer.append(transition()); buffer.append(transition('bus','BUS_ARRIVAL'))
    assert len(buffer)==2 and len(buffer.by_agent('assignment'))==1 and len(buffer.by_agent('bus'))==1

def test_buffer_does_not_fabricate_inactive_agents():
    buffer=AsyncMAPPOBuffer(); buffer.append(transition())
    assert [item.agent_id for item in buffer.transitions]==['assignment'] and buffer.by_agent('bus')==[]

def test_returns_normalize_and_clear():
    buffer=AsyncMAPPOBuffer(); buffer.append(transition()); buffer.transitions[0].done=True
    returns,advantages=buffer.compute_returns_and_advantages(.99,.95)
    assert returns.shape==(1,) and np.isfinite(advantages).all()
    buffer.clear(); assert len(buffer)==0

def test_invalid_agent_rejected():
    with pytest.raises(ValueError): AsyncMAPPOBuffer().append(transition('inactive'))

def test_buffer_accepts_four_agent_types():
    buffer = AsyncMAPPOBuffer()
    for agent, event in (
        ("assignment", "PARCEL_RELEASE"),
        ("truck", "TRUCK_AVAILABLE"),
        ("bus", "BUS_DEPARTURE"),
        ("station", "STATION_OPERATION"),
    ):
        buffer.append(transition(agent=agent, event=event))
    assert [item.agent_id for item in buffer.transitions] == ["assignment", "truck", "bus", "station"]

def test_event_time_discount_uses_elapsed_minutes():
    buffer = AsyncMAPPOBuffer()
    first = transition(agent="assignment", event="PARCEL_RELEASE", episode=0)
    second = transition(agent="truck", event="TRUCK_AVAILABLE", episode=0)
    first.event_time = 0.0
    second.event_time = 10.0
    first.reward = 0.0
    first.value = 0.0
    second.reward = 1.0
    second.value = 0.0
    second.done = True
    buffer.append(first)
    buffer.append(second)
    returns, _advantages = buffer.compute_returns_and_advantages(0.9, 1.0, reference_time_unit=10.0)
    assert returns[0] == pytest.approx(0.9)
    assert returns[1] == pytest.approx(1.0)
