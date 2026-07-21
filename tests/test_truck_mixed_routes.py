from tests.test_truck_batch_generation import env,p
from envs.action_generators.truck_batch_actions import generate_truck_batch_candidates
from types import SimpleNamespace

def test_mixed_direct_terminal_station_and_multiple_stations():
    parcels=[p("p1","TD"),p("p2","TBD",st="s1"),p("p3","TLD",st="s1"),p("p4","TLD",st="s2")]
    e=env(parcels); t=SimpleNamespace(truck_id="t",current_location_id="depot_01",onboard_parcels=[],available_time=0)
    c=next(c for c in generate_truck_batch_candidates(e,t) if len(c.parcel_ids)==4)
    types=[s.stop_type for s in c.ordered_route_stops]
    assert {"CUSTOMER","BUS_TERMINAL","INTEGRATED_STATION"} <= set(types)
    assert len([s for s in c.ordered_route_stops if s.stop_type=="INTEGRATED_STATION"])==2
