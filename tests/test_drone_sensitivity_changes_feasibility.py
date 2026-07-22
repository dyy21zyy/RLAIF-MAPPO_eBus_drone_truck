from types import SimpleNamespace as N
import numpy as np
from envs.action_generators.station_actions import generate_station_operation_candidates
from envs.dynamics.station_power import StationBaseLoadInterval, StationBaseLoadProfile


def make(payload=5, radius=8, max_rt=120, dist_km=7, speed=8.4):
    drone=N(drone_id='d', status='AVAILABLE', available_time_min=0)
    bat=N(battery_id='b', status='FULL')
    st=N(station_id='s', active_bus_charges=[], active_battery_charges=[], battery_power_kw=2, power_capacity_kw=1000, charging_slots=1, drone_states=[drone], battery_states=[bat])
    p=N(parcel_id='p', status='WAITING_DRONE', station_id='s', weight_kg=4, deadline_min=999, priority=1)
    return N(now_min=0, config={'bus':{'charging_power_kw':500}, 'station':{'max_operation_candidates':8}, 'network':{'drone_speed_kmph':speed,'drone_payload_kg':payload,'drone_radius_km':radius,'max_drone_round_trip_min':max_rt}}, stations={'s':st}, parcels={'p':p}, waiting_station_parcels={'s':['p']}, drone_distance_m=np.array([[dist_km*1000]]), drone_row_index={'s':0}, drone_column_index={'p':0}, station_base_load_profile=StationBaseLoadProfile([StationBaseLoadInterval('s','a',0,200,0)]))


def has_dispatch(e):
    return any(c.dispatches for c in generate_station_operation_candidates(e,'s'))


def test_payload_radius_and_duration_change_feasibility():
    assert has_dispatch(make(payload=5))
    assert not has_dispatch(make(payload=3))
    assert has_dispatch(make(radius=8))
    assert not has_dispatch(make(radius=6))
    assert has_dispatch(make(max_rt=120))
    assert not has_dispatch(make(max_rt=90))
