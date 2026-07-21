from types import SimpleNamespace as N
from envs.action_generators.bus_loading_actions import generate_bus_loading_candidates, _make

def p(pid,station,w): return N(parcel_id=pid,status="AT_BUS_TERMINAL",mode="TBD",station_id=station,weight_kg=w,deadline_min=99,priority=1)
def env(ps): return N(now_min=0,parcels={x.parcel_id:x for x in ps},trip_stop_times={"t":[{"stop_id":"a"},{"stop_id":"b"}]},stop_to_station={"b":"s1"},pending_bus_parcels={},physical_buses={"bus":N(onboard_parcel_ids=[],passenger_manifest=N(total_onboard_passengers=0))},trip_to_bus={"t":"bus"},bus_freight_kg={"t":0},config={"bus":{"freight_capacity_kg":20,"terminal_loading_time_min_per_kg":0}})

def test_capacity_station_limit_and_downstream_are_hard():
    e=env([p("a","s1",11),p("b","s1",10),p("c","s2",1)])
    assert not _make(e,"t","bus",["a"],"x").feasible
    assert "station_unload_limit_exceeded" in _make(e,"t","bus",["a"],"x").infeasibility_reasons
    assert not _make(e,"t","bus",["a","b"],"x").feasible
    assert "bus_freight_capacity_exceeded" in _make(e,"t","bus",["a","b"],"x").infeasibility_reasons
    assert "target_not_downstream" in _make(e,"t","bus",["c"],"x").infeasibility_reasons
    assert all(c.feasible for c in generate_bus_loading_candidates(e,"t"))
