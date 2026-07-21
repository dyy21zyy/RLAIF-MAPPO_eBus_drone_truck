from training.mappo_async import VALID_AGENT_EVENTS

def test_only_active_agent_event_pairs_are_registered():
    assert {"PARCEL_RELEASE"} <= VALID_AGENT_EVENTS["assignment"]
    assert {"TRUCK_AVAILABLE"} <= VALID_AGENT_EVENTS["truck"]
    assert {"BUS_TERMINAL_DEPARTURE","BUS_STATION_ARRIVAL"} <= VALID_AGENT_EVENTS["bus"]
    assert {"STATION_OPERATION"} <= VALID_AGENT_EVENTS["station"]
