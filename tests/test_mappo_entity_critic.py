from types import SimpleNamespace
from training.entity_encoders import encode_entity_critic_state

def env(n_parcels, n_stations, n_trucks, n_buses):
    return SimpleNamespace(now_min=1,horizon_min=100,config={"bus":{"bus_battery_kwh":100}},events=[],parcels={str(i):SimpleNamespace(status="DELIVERED",deadline_min=50,delivered_time_min=40) for i in range(n_parcels)},trucks=[SimpleNamespace(available_time=0,onboard_parcels=[],total_travel_time=0) for _ in range(n_trucks)],physical_buses={str(i):SimpleNamespace(soc_kwh=50,schedule_delay_min=0,passenger_manifest=SimpleNamespace(total_onboard_passengers=0)) for i in range(n_buses)},passenger_stops={},stations={str(i):SimpleNamespace(locker_load_kg=0,locker_capacity_kg=10,drone_states=[],battery_states=[],charging_slots=1) for i in range(n_stations)})

def test_critic_input_dimension_fixed_across_instance_sizes():
    assert len(encode_entity_critic_state(env(2,1,1,1))) == len(encode_entity_critic_state(env(5,3,2,4)))
