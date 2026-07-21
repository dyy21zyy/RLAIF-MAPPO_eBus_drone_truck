from types import SimpleNamespace as N
from copy import deepcopy
from envs.action_generators.bus_loading_actions import generate_bus_loading_candidates

def p(pid,station,w,deadline=50,priority=1): return N(parcel_id=pid,status="AT_BUS_TERMINAL",mode="TBD",station_id=station,weight_kg=w,deadline_min=deadline,priority=priority)
def e():
    ps=[p("a","s1",2,30,1),p("b","s1",3,20,3),p("c","s2",4,40,2),p("d","s2",5,10,1)]
    return N(now_min=0,horizon_min=100,station_ids=["s1","s2"],parcels={x.parcel_id:x for x in ps},trip_stop_times={"t":[{"stop_id":"term"},{"stop_id":"x"},{"stop_id":"y"}]},stop_to_station={"x":"s1","y":"s2"},pending_bus_parcels={},physical_buses={"b":N(onboard_parcel_ids=[],passenger_manifest=N(total_onboard_passengers=0))},trip_to_bus={"t":"b"},bus_freight_kg={"t":0},config={"bus":{"freight_capacity_kg":20,"terminal_loading_time_min_per_kg":0.1,"station_unloading_time_min_per_kg":0.1}})

def test_multiple_distinct_loading_batches_and_side_effect_free():
    env=e(); before=deepcopy(env.parcels); c=generate_bus_loading_candidates(env,"t")
    assert len({x.parcel_ids for x in c})>2
    assert env.parcels==before
    assert {x.heuristic_source for x in c} >= {"earliest-deadline-first","highest-priority-first","maximum-weight-utilization"}
