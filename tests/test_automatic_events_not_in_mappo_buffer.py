import pytest
from training.mappo_buffer import AsyncMAPPOBuffer, AsyncTransition

def _t(event="PARCEL_RELEASE", agent="assignment", eid=0):
    return AsyncTransition(agent, [0.], [0.], 0, [True], [[0.]], ("x",), 0., 0., 0., False, [0.], event, 0., eid, environment_reward=0., total_reward=0.)

def test_automatic_events_cannot_be_appended():
    with pytest.raises(ValueError): AsyncMAPPOBuffer().append(_t("BUS_TRIP_START","bus",2))
