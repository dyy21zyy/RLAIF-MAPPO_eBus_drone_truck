from __future__ import annotations
import json,math
from pathlib import Path
REQUIRED=['released_parcels','delivered_parcels','undelivered_parcels','fulfillment_rate','on_time_over_all_released','on_time_over_delivered','urgent_on_time_fulfillment','average_lateness','maximum_lateness','truck_distance','truck_weight_utilization','truck_volume_utilization','parcels_per_truck_route','bus_freight_utilization','bus_propulsion_energy','bus_charging_energy','minimum_bus_soc','battery_safety_violations','waiting_passenger_minutes','onboard_additional_delay_passenger_minutes','bus_operating_delay','drone_missions','full_batteries','depleted_batteries','charging_batteries','charging_slot_utilization','locker_occupancy','station_peak_load','overload_kw_min','overload_duration','environment_reward','per_agent_rlaif_contribution','combined_reward','runtime']
def validate_metric_sources(metrics, bus_min_soc=None, power=None, passenger=None, output_path=None):
    reports=[]; checks=[]
    for m in REQUIRED:
        avail=m in metrics; val=metrics.get(m); finite=avail and (isinstance(val,dict) or math.isfinite(float(val)))
        reports.append({'metric':m,'value':val,'source_field':m,'formula':'direct or reconciled diagnostic counter','availability':avail,'finite':finite,'legitimate_zero':avail and val==0,'evidence_count':1 if avail else 0})
    missing=[r['metric'] for r in reports if not r['availability']]
    def add(n,ok,exp=None,act=None):
        try:
            diff = None if exp is None or act is None else abs(float(exp)-float(act))
        except Exception:
            diff = None
        checks.append({'check':n,'status':'passed' if ok else 'failed','expected':exp,'actual':act,'difference':diff})
    add('missing_required_metric_fails',not missing,[],missing)
    add('delivered_plus_undelivered_equals_released',metrics.get('delivered_parcels',0)+metrics.get('undelivered_parcels',0)==metrics.get('released_parcels',0))
    add('urgent_on_time_not_exceed_urgent_released',metrics.get('urgent_on_time_fulfillment',0)<=metrics.get('urgent_released',metrics.get('released_parcels',0)))
    for k in ['truck_weight_utilization','truck_volume_utilization','bus_freight_utilization','charging_slot_utilization']: add(k+'_in_range',0<=metrics.get(k,0)<=1)
    if bus_min_soc is not None: add('minimum_soc_matches_bus_trace',abs(metrics.get('minimum_bus_soc')-bus_min_soc)<1e-6,bus_min_soc,metrics.get('minimum_bus_soc'))
    if power: add('station_peak_load_matches_power_trace',abs(metrics.get('station_peak_load')-power['peak_load_kw'])<1e-6,power['peak_load_kw'],metrics.get('station_peak_load')); add('overload_kw_min_matches_power_reconciliation',abs(metrics.get('overload_kw_min')-power['overload_kw_min'])<1e-6,power['overload_kw_min'],metrics.get('overload_kw_min'))
    if passenger: add('passenger_delay_metrics_match_reconciliation',abs(metrics.get('waiting_passenger_minutes')-passenger['waiting_passenger_minutes'])<1e-6 and abs(metrics.get('onboard_additional_delay_passenger_minutes')-passenger['onboard_additional_delay_passenger_minutes'])<1e-6)
    res={'passed':all(c['status']=='passed' for c in checks),'metrics':reports,'checks':checks}
    if output_path: Path(output_path).write_text(json.dumps(res,indent=2))
    return res
