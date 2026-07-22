import pytest
from training.mappo_buffer import AsyncMAPPOBuffer, AsyncTransition

def t(agent="bus", event="BUS_STATION_ARRIVAL", eid=3):
    return AsyncTransition(agent,[0.],[0.],0,[True],[[0.]],("x",),0.,0.,1.,False,[0.],event,0.,eid,environment_reward=1.,learned_reward_weighted=0.,total_reward=1.)

def test_buffer_stores_canonical_event_id_and_rejects_mismatch():
    b=AsyncMAPPOBuffer(); b.append(t(event="BUS_ARRIVAL", eid=3)); assert b.transitions[0].event_type=="BUS_STATION_ARRIVAL" and b.transitions[0].event_type_id==3
    with pytest.raises(ValueError): b.append(t(agent="truck", event="BUS_STATION_ARRIVAL", eid=3))
