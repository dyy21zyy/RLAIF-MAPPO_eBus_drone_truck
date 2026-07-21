from types import SimpleNamespace
from evaluation.metrics import collect_formal_runtime_metrics

def env_with(parcels):
    return SimpleNamespace(parcels=parcels,trucks=[],truck_dispatch_count=0,truck_weight_utilization_sum=0,truck_volume_utilization_sum=0,truck_parcels_routed=0,bus_freight_utilization=0,bus_propulsion_energy_kwh=0,bus_charging_energy_kwh=0,bus_soc_kwh={'b':10},battery_safety_violation_count=0,passenger_waiting_minutes=0,passenger_onboard_delay_minutes=0,raw_cost_components={'bus_operating_delay':0},drone_mission_count=0,charging_slot_busy_minutes=0,charging_slot_available_minutes=1,locker_occupancy_kg_minutes=0,peak_station_load_kw=0,accumulated_power_overload=0)

def test_urgent_metric_uses_is_urgent_not_priority():
    p={'a':SimpleNamespace(release_time_min=0,status='DELIVERED',delivered_time_min=4,deadline_min=5,is_urgent=True,priority=1),'b':SimpleNamespace(release_time_min=0,status='DELIVERED',delivered_time_min=4,deadline_min=5,is_urgent=False,priority=99)}
    assert collect_formal_runtime_metrics(env_with(p)).urgent_on_time_fulfillment == 1.0
