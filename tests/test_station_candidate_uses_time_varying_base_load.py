from types import SimpleNamespace as N
from envs.action_generators.station_actions import projected_load, generate_station_operation_candidates
from envs.dynamics.station_power import StationBaseLoadInterval, StationBaseLoadProfile


def env_at(t, cap=150):
    st=N(station_id='s', active_bus_charges=[], active_battery_charges=[], battery_power_kw=2, power_capacity_kw=cap, charging_slots=1, drone_states=[], battery_states=[])
    return N(now_min=t, config={'bus':{'charging_power_kw':500}, 'station':{'max_operation_candidates':4}, 'network':{'drone_speed_kmph':40,'drone_payload_kg':5,'drone_radius_km':8,'max_drone_round_trip_min':120}}, stations={'s':st}, parcels={}, waiting_station_parcels={'s':[]}, station_base_load_profile=StationBaseLoadProfile([StationBaseLoadInterval('s','a',0,10,100), StationBaseLoadInterval('s','b',10,20,120)])), st


def test_projected_load_uses_current_profile_interval():
    e, st=env_at(5)
    assert projected_load(e, st, 0) == 100
    e.now_min=15
    assert projected_load(e, st, 0) == 120


def test_candidate_power_fields_are_component_sum_and_soft_capacity():
    e, st=env_at(5, cap=90)
    c=generate_station_operation_candidates(e,'s')[0]
    assert c.projected_station_load == 100
    assert c.projected_overload == 10
    assert c.power_margin == -10
    assert c.feasible
