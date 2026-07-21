from types import SimpleNamespace
import numpy as np
from envs.action_generators.station_actions import generate_station_operation_candidates
from envs.dynamics.station_dynamics import dispatch_drone
from envs.delivery_env import RuntimeDroneState, RuntimeBatteryState, StationState, ParcelState

def env():
    e=SimpleNamespace(now_min=0.0,horizon_min=200.0,config={"network":{"drone_speed_kmph":60,"customer_service_time_min":1,"drone_turnaround_time_min":1},"station":{"max_operation_candidates":16,"base_load_kw":0}},events=[])
    e._push=lambda t,k,p: e.events.append((t,k,p))
    e.drone_row_index={"s":0}; e.drone_column_index={f"p{i}":i for i in range(3)}; e.drone_distance_m=np.array([[1000,500,1500]])
    e.stations={"s":StationState("s","stop",100,3,3,1100,2,45)}
    st=e.stations["s"]
    st.drone_states=[RuntimeDroneState(f"d{i}","s") for i in range(3)]
    st.battery_states=[RuntimeBatteryState(f"b{i}","s","FULL") for i in range(3)]
    e.parcels={f"p{i}":ParcelState(f"p{i}",0,50-i*10,1,0.1,i+1,"s",True,status="WAITING_DRONE",station_id="s") for i in range(3)}
    e.waiting_station_parcels={"s":["p0","p1","p2"]}
    return e

def test_station_can_select_non_first_parcel_and_multiple_drones():
    c=generate_station_operation_candidates(env(),"s")
    assert any(any(p!="p0" for _,p,_ in x.dispatches) for x in c)
    assert any(len(x.dispatches) >= 2 for x in c)

def test_one_drone_or_parcel_not_duplicated_in_candidate():
    for c in generate_station_operation_candidates(env(),"s"):
        ds=[d for d,_,_ in c.dispatches]; ps=[p for _,p,_ in c.dispatches]
        assert len(ds)==len(set(ds)); assert len(ps)==len(set(ps))
