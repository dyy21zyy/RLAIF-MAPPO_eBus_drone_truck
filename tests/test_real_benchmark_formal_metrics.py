import pytest
from evaluation.formal_metric_validation import validate_formal_metrics, MissingFormalMetricError, NonFiniteFormalMetricError, FormalMetricReconciliationError

def row(v=0.0):
    names=['fulfillment_rate','on_time_over_all_released','on_time_over_delivered','urgent_on_time_fulfillment','average_lateness','maximum_lateness','undelivered_parcels','truck_distance','truck_weight_utilization','truck_volume_utilization','parcels_per_truck_route','bus_freight_utilization','bus_propulsion_energy','bus_charging_energy','minimum_bus_soc','battery_safety_violations','waiting_passenger_minutes','onboard_additional_delay_passenger_minutes','bus_operating_delay','drone_missions','full_battery_availability','depleted_battery_inventory','charging_slot_utilization','locker_occupancy','station_peak_load','overload_kw_min','overload_duration','environment_reward','rlaif_total_weighted','combined_reward_total','runtime']
    names += [f'rlaif_{a}_{k}' for a in ('assignment','truck','bus','station') for k in ('raw','weighted')]
    return {n:{'value':v,'availability':'available','source':'test','legitimate_zero':True} for n in names}

def test_legitimate_zero_passes(): validate_formal_metrics(row())
def test_missing_required_metric_fails():
    r=row(); r.pop('runtime')
    with pytest.raises(MissingFormalMetricError): validate_formal_metrics(r)
def test_nonfinite_metric_fails():
    r=row(); r['runtime']['value']=float('nan')
    with pytest.raises(NonFiniteFormalMetricError): validate_formal_metrics(r)
def test_reward_totals_reconcile():
    r=row(); r['rlaif_assignment_weighted']['value']=1; r['rlaif_total_weighted']['value']=2
    with pytest.raises(FormalMetricReconciliationError): validate_formal_metrics(r)
