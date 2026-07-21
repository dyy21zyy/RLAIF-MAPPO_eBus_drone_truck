from types import SimpleNamespace
import numpy as np
from envs.action_generators.truck_batch_actions import generate_truck_batch_candidates

def env(parcels):
    loc=["depot_01","p1","p2","terminal","s1","s2"]
    idx={x:i for i,x in enumerate(loc)}; n=len(loc)
    m=np.abs(np.subtract.outer(range(n),range(n))).astype(float)*1000
    return SimpleNamespace(now_min=0,horizon_min=500,parcels={p.parcel_id:p for p in parcels},trucks=[],truck_location_index=idx,truck_distance_m=m,truck_time_min=m/1000*10,trip_stop_times={"trip":[{"stop_id":"terminal"}]},config={"truck":{"weight_capacity_kg":10,"volume_capacity_m3":10,"max_batch_size":4,"max_batch_candidates":6,"return_to_depot":True}})
def p(i,mode="TD",w=1,v=1,st="s1"):
    return SimpleNamespace(parcel_id=i,mode=mode,weight_kg=w,volume=v,status="WAITING_TRUCK",release_time_min=0,deadline_min=100,priority=1,station_id=st)
def test_one_action_can_select_multiple_and_is_side_effect_free_bounded_dedup_idle():
    e=env([p("p1"),p("p2")]); t=SimpleNamespace(truck_id="t",current_location_id="depot_01",onboard_parcels=[],available_time=0)
    before=[x.status for x in e.parcels.values()]
    cs=generate_truck_batch_candidates(e,t)
    assert any(c.idle_flag for c in cs)
    assert any(len(c.parcel_ids)>=2 for c in cs)
    assert [x.status for x in e.parcels.values()]==before
    assert len(cs)<=6
    sigs={(c.parcel_ids, tuple((s.stop_id,s.stop_type,s.parcel_ids) for s in c.ordered_route_stops)) for c in cs}
    assert len(sigs)==len(cs)
def test_different_direct_customers_in_one_batch():
    e=env([p("p1"),p("p2")]); t=SimpleNamespace(truck_id="t",current_location_id="depot_01",onboard_parcels=[],available_time=0)
    c=next(c for c in generate_truck_batch_candidates(e,t) if set(c.parcel_ids)=={"p1","p2"})
    assert c.number_of_direct_customers==2
