from types import SimpleNamespace as N
from envs.action_generators.bus_loading_actions import eligible_parcel_ids

def p(pid,status="AT_BUS_TERMINAL",mode="TBD",station="s1",w=1,arr=0):
    return N(parcel_id=pid,status=status,mode=mode,station_id=station,weight_kg=w,deadline_min=99,priority=1,truck_terminal_arrival_min=arr)
def env(parcels):
    return N(now_min=10,parcels={x.parcel_id:x for x in parcels},trip_stop_times={"t":[{"stop_id":"terminal"},{"stop_id":"stop1"}]},stop_to_station={"stop1":"s1"},pending_bus_parcels={},physical_buses={"b":N(onboard_parcel_ids=[])})

def test_only_terminal_tbd_are_eligible_and_unreleased_excluded():
    e=env([p("ok"),p("bad_status","UNRELEASED"),p("bad_mode","AT_BUS_TERMINAL","TD")])
    assert eligible_parcel_ids(e,"t","b")==["ok"]

def test_parcels_arriving_after_departure_cutoff_cannot_load():
    e=env([p("early",arr=9),p("late",arr=11)])
    assert eligible_parcel_ids(e,"t","b")==["early"]
