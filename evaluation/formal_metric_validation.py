"""Fail-closed formal metric validation with explicit legitimate-zero handling."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import math

REQUIRED_FORMAL_METRICS = (
 'fulfillment_rate','on_time_over_all_released','on_time_over_delivered','urgent_on_time_fulfillment','average_lateness','maximum_lateness','undelivered_parcels',
 'truck_distance','truck_weight_utilization','truck_volume_utilization','parcels_per_truck_route','bus_freight_utilization','bus_propulsion_energy','bus_charging_energy','minimum_bus_soc','battery_safety_violations',
 'waiting_passenger_minutes','onboard_additional_delay_passenger_minutes','bus_operating_delay','drone_missions','full_battery_availability','depleted_battery_inventory','charging_slot_utilization','locker_occupancy',
 'station_peak_load','overload_kw_min','overload_duration','environment_reward','rlaif_total_weighted','combined_reward_total','runtime')
RLAIF_FIELDS=tuple(f'rlaif_{a}_{kind}' for a in ('assignment','truck','bus','station') for kind in ('raw','weighted'))

@dataclass(frozen=True)
class MetricRecord:
    value: float|int
    availability: str
    source: str
    legitimate_zero: bool
    finite: bool

class FormalMetricValidationError(RuntimeError): pass
class MissingFormalMetricError(FormalMetricValidationError): pass
class NonFiniteFormalMetricError(FormalMetricValidationError): pass
class FormalMetricReconciliationError(FormalMetricValidationError): pass

def metric_source_map(metrics:dict[str,Any])->dict[str,str]:
    return {k:(v.get('source') if isinstance(v,dict) else k) for k,v in metrics.items()}

def _coerce(name:str, obj:Any)->MetricRecord:
    if isinstance(obj,MetricRecord): return obj
    if isinstance(obj,dict):
        if obj.get('availability') == 'missing': raise MissingFormalMetricError(f'missing required metric: {name}')
        if 'value' not in obj: raise MissingFormalMetricError(f'missing value for metric: {name}')
        val=obj['value']; finite=math.isfinite(float(val))
        return MetricRecord(val, obj.get('availability','available'), obj.get('source',name), bool(obj.get('legitimate_zero', float(val)==0.0)), finite)
    if obj is None: raise MissingFormalMetricError(f'missing required metric: {name}')
    finite=math.isfinite(float(obj))
    return MetricRecord(obj,'available',name,bool(float(obj)==0.0),finite)

def validate_formal_metrics(row:dict[str,Any], *, fail_on_missing:bool=True)->dict[str,dict[str,Any]]:
    required=REQUIRED_FORMAL_METRICS + RLAIF_FIELDS
    out={}
    for name in required:
        if name not in row:
            if fail_on_missing: raise MissingFormalMetricError(f'missing required metric: {name}')
            continue
        rec=_coerce(name,row[name])
        if not rec.finite: raise NonFiniteFormalMetricError(f'nonfinite metric: {name}')
        if float(rec.value)==0.0 and not rec.legitimate_zero: raise FormalMetricValidationError(f'zero metric lacks legitimate-zero provenance: {name}')
        out[name]=asdict(rec)
    total=sum(float(out[f'rlaif_{a}_weighted']['value']) for a in ('assignment','truck','bus','station'))
    if abs(total-float(out['rlaif_total_weighted']['value']))>1e-9: raise FormalMetricReconciliationError('RLAIF total does not reconcile')
    combined=float(out['environment_reward']['value'])+float(out['rlaif_total_weighted']['value'])
    if abs(combined-float(out['combined_reward_total']['value']))>1e-9: raise FormalMetricReconciliationError('combined reward does not reconcile')
    out['metric_source_map']={k:v['source'] for k,v in out.items()}
    return out
