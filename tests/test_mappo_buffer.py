"""Asynchronous MAPPO buffer tests."""
import numpy as np
import pytest
from training.mappo_buffer import AsyncMAPPOBuffer, AsyncTransition

def transition(agent='assignment', event='PARCEL_ARRIVAL', episode=0):
    return AsyncTransition(agent,[0.0],[0.0],0,[True],-0.2,0.1,1.0,False,[0.1],event,2.0,episode_id=episode)

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
