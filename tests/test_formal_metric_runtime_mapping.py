from types import SimpleNamespace
from evaluation.metrics import collect_formal_runtime_metrics

def test_formal_metrics_map_real_counters():
    env=SimpleNamespace(parcels={},trucks=[SimpleNamespace(total_distance=12)],truck_dispatch_count=2,truck_weight_utilization_sum=1.0,truck_volume_utilization_sum=0.5,truck_parcels_routed=6,bus_freight_utilization=0.3,bus_propulsion_energy_kwh=4,bus_charging_energy_kwh=5,bus_soc_kwh={'x':7,'y':9},battery_safety_violation_count=1,passenger_waiting_minutes=11,passenger_onboard_delay_minutes=13,raw_cost_components={'bus_operating_delay':17},drone_mission_count=3,charging_slot_busy_minutes=2,charging_slot_available_minutes=4,locker_occupancy_kg_minutes=19,peak_station_load_kw=20,accumulated_power_overload=21)
    m=collect_formal_runtime_metrics(env)
    assert m.truck_weight_utilization==0.5 and m.truck_volume_utilization==0.25 and m.parcels_per_truck_route==3
    assert m.waiting_passenger_minutes==11 and m.onboard_additional_delay_passenger_minutes==13 and m.minimum_bus_soc==7
