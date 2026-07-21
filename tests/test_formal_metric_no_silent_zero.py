from types import SimpleNamespace
import pytest
from evaluation.metrics import FormalMetricError, collect_formal_runtime_metrics

def test_missing_required_metric_raises_but_legitimate_zero_allowed():
    with pytest.raises(FormalMetricError): collect_formal_runtime_metrics(SimpleNamespace(parcels={}))
    env=SimpleNamespace(parcels={},trucks=[],truck_dispatch_count=0,truck_weight_utilization_sum=0,truck_volume_utilization_sum=0,truck_parcels_routed=0,bus_freight_utilization=0,bus_propulsion_energy_kwh=0,bus_charging_energy_kwh=0,bus_soc_kwh={'x':0},battery_safety_violation_count=0,passenger_waiting_minutes=0,passenger_onboard_delay_minutes=0,raw_cost_components={'bus_operating_delay':0},drone_mission_count=0,charging_slot_busy_minutes=0,charging_slot_available_minutes=1,locker_occupancy_kg_minutes=0,peak_station_load_kw=0,accumulated_power_overload=0)
    assert collect_formal_runtime_metrics(env).bus_propulsion_energy_kwh == 0
