from types import SimpleNamespace as N
from envs.action_generators.station_actions import generate_station_operation_candidates
from envs.dynamics.station_power import StationBaseLoadInterval, StationBaseLoadProfile


def test_charging_slots_and_capacity_from_runtime_station():
    bat=N(battery_id='b', status='DEPLETED')
    st=N(station_id='s', active_bus_charges=[], active_battery_charges=[], battery_power_kw=7, power_capacity_kw=5, charging_slots=0, drone_states=[], battery_states=[bat])
    e=N(now_min=0, config={'bus':{'charging_power_kw':500}, 'station':{'max_operation_candidates':8}, 'network':{'drone_speed_kmph':40,'drone_payload_kg':5,'drone_radius_km':8,'max_drone_round_trip_min':120}}, stations={'s':st}, parcels={}, waiting_station_parcels={'s':[]}, station_base_load_profile=StationBaseLoadProfile([StationBaseLoadInterval('s','a',0,10,4)]))
    cs=generate_station_operation_candidates(e,'s')
    assert not any(c.battery_ids_to_start_charging for c in cs)
    assert cs[0].projected_station_load == 4
    assert cs[0].power_margin == 1
    st.charging_slots=1
    cs=generate_station_operation_candidates(e,'s')
    charge=[c for c in cs if c.battery_ids_to_start_charging][0]
    assert charge.projected_station_load == 11
    assert charge.projected_overload == 6
    assert charge.feasible
