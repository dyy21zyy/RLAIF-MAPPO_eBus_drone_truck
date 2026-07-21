from tests.test_truck_batch_generation import env,p
from envs.action_generators.truck_batch_actions import generate_truck_batch_candidates
from envs.dynamics.truck_dynamics import apply_truck_batch
from types import SimpleNamespace

def test_fixed_dispatch_charged_once_and_utilization_metrics():
    e=env([p("p1"),p("p2")]); e.config["truck"].update({"fixed_dispatch_cost":5,"cost_per_km":1,"cost_per_min":0}); e.config["reward"]={"truck_cost":1}
    e.cost_components={"truck_cost":0}; e._charge_cost=lambda k,a: e.cost_components.__setitem__(k,e.cost_components.get(k,0)+a) or -a
    e.events=[]; e.event_sequence=0; e._push=lambda *args: e.events.append(args)
    t=SimpleNamespace(truck_id="t",current_location_id="depot_01",onboard_parcels=[],available_time=0,status="idle",remaining_capacity_kg=10,total_distance=0,total_travel_time=0,route_history=[])
    e.trucks=[t]
    c=next(c for c in generate_truck_batch_candidates(e,t) if len(c.parcel_ids)==2)
    apply_truck_batch(e,t,c)
    assert e.cost_components["truck_cost"] == 5 + c.estimated_distance_km
    assert e.truck_dispatch_count == 1 and e.truck_parcels_routed == 2
    assert e.truck_weight_utilization_sum == c.weight_utilization
