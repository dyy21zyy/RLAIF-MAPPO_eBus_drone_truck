import pytest
from training.event_schema import decision_event_id, decision_event_agent, normalize_decision_event_type, is_decision_event

def test_canonical_event_ids_and_agents():
    expected={"PARCEL_RELEASE":(0,"assignment"),"TRUCK_AVAILABLE":(1,"truck"),"BUS_TERMINAL_DEPARTURE":(2,"bus"),"BUS_STATION_ARRIVAL":(3,"bus"),"STATION_OPERATION":(4,"station")}
    for name,(eid,agent) in expected.items():
        assert decision_event_id(name)==eid
        assert decision_event_agent(name)==agent

def test_legacy_and_unknown_and_automatic():
    assert normalize_decision_event_type("BUS_DEPARTURE")=="BUS_TERMINAL_DEPARTURE"
    assert normalize_decision_event_type("BUS_ARRIVAL")=="BUS_STATION_ARRIVAL"
    assert not is_decision_event("BUS_TRIP_START")
    with pytest.raises(ValueError): normalize_decision_event_type("NOPE")
