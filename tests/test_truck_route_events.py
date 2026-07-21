from tests.test_truck_batch_generation import env,p
from envs.action_generators.truck_batch_actions import generate_truck_batch_candidates
from envs.dynamics.truck_dynamics import apply_truck_batch
from types import SimpleNamespace

def test_route_events_and_selection_state_changes_only_on_action():
    e=env([p("p1","TD"),p("p2","TBD",st="s1"),p("p3","TLD",st="s1")]); e.config["reward"]={"truck_cost":1}; e.cost_components={"truck_cost":0}; e._charge_cost=lambda k,a: -a
    e.events=[]; e.event_sequence=0; e._push=lambda time,kind,payload: e.events.append((time,kind,payload))
    t=SimpleNamespace(truck_id="t",current_location_id="depot_01",onboard_parcels=[],available_time=0,status="idle",remaining_capacity_kg=10,total_distance=0,total_travel_time=0,route_history=[])
    e.trucks=[t]
    c=next(c for c in generate_truck_batch_candidates(e,t) if len(c.parcel_ids)==3)
    assert all(x.status=="WAITING_TRUCK" for x in e.parcels.values())
    apply_truck_batch(e,t,c)
    assert all(x.status=="ONBOARD_TRUCK" for x in e.parcels.values())
    kinds=[k for _,k,_ in e.events]
    assert ["truck_departure","truck_arrive_stop","truck_unload"] == kinds[:3]
    assert "truck_route_complete" in kinds and "truck_available" in kinds
    assert t.status=="traveling" and t.available_time>0
