"""Metric extraction from the Stage 3 environment without optional dependencies."""
from __future__ import annotations
from dataclasses import dataclass
from statistics import fmean
from envs.status import is_delivered_status

class FormalMetricError(RuntimeError):
    pass

@dataclass(frozen=True)
class FormalRuntimeMetrics:
    fulfillment_rate: float
    on_time_over_all_released: float
    on_time_over_delivered: float
    urgent_on_time_fulfillment: float
    average_lateness: float
    maximum_lateness: float
    undelivered_parcels: int
    truck_distance: float
    truck_weight_utilization: float
    truck_volume_utilization: float
    parcels_per_truck_route: float
    bus_freight_utilization: float
    bus_propulsion_energy_kwh: float
    bus_charging_energy_kwh: float
    minimum_bus_soc: float
    battery_safety_violations: int
    waiting_passenger_minutes: float
    onboard_additional_delay_passenger_minutes: float
    bus_operating_delay: float
    drone_missions: int
    charging_slot_utilization: float
    locker_occupancy: float
    station_peak_power: float
    overload_kw_min: float


FORMAL_METRICS = (
    "total_normalized_cost", "environment_reward", "rlaif_reward_assignment", "rlaif_reward_bus", "rlaif_reward_truck", "rlaif_reward_station",
    "fulfillment_rate", "on_time_over_all_released", "on_time_over_delivered", "urgent_on_time_fulfillment", "average_lateness", "maximum_lateness",
    "undelivered_parcels", "truck_distance", "truck_weight_utilization", "truck_volume_utilization", "parcels_per_truck_route", "bus_freight_utilization",
    "bus_propulsion_energy_kwh", "bus_charging_energy_kwh", "bus_energy", "minimum_bus_soc", "battery_safety_violations", "waiting_passenger_minutes", "onboard_additional_delay_passenger_minutes", "bus_operating_delay",
    "drone_missions", "charging_slot_utilization", "locker_occupancy", "station_peak_power", "overload_kw_min", "runtime"
)

FORMAL_METRIC_SOURCE_MAP={
 'truck_weight_utilization':'truck_weight_utilization_sum / truck_dispatch_count', 'truck_volume_utilization':'truck_volume_utilization_sum / truck_dispatch_count', 'parcels_per_truck_route':'truck_parcels_routed / truck_dispatch_count', 'waiting_passenger_minutes':'passenger_waiting_minutes', 'onboard_additional_delay_passenger_minutes':'passenger_onboard_delay_minutes'}

def _require(env, name):
    if not hasattr(env, name): raise FormalMetricError(f"missing required runtime metric source: {name}")
    return getattr(env, name)

def _ratio(num, den):
    return float(num) / float(den) if float(den) else 0.0

def collect_formal_runtime_metrics(env) -> FormalRuntimeMetrics:
    parcels=list(_require(env,'parcels').values())
    released=[p for p in parcels if getattr(p,'release_time_min',None) is not None]
    delivered=[p for p in released if is_delivered_status(getattr(p,'status','')) and getattr(p,'delivered_time_min',None) is not None]
    urgent=[p for p in released if bool(_require(p,'is_urgent'))]
    on_time=[p for p in delivered if float(p.delivered_time_min)<=float(p.deadline_min)]
    urgent_on_time=[p for p in delivered if bool(_require(p,'is_urgent')) and float(p.delivered_time_min)<=float(p.deadline_min)]
    lateness=[max(0.0,float(p.delivered_time_min)-float(p.deadline_min)) for p in delivered]
    dispatch=_require(env,'truck_dispatch_count')
    bus_soc=getattr(env,'bus_soc_kwh',{})
    if not bus_soc and hasattr(env,'physical_buses'):
        bus_soc={k:getattr(v,'soc_kwh') for k,v in env.physical_buses.items()}
    if not bus_soc: raise FormalMetricError('missing required runtime metric source: bus_soc_kwh/physical_buses')
    busy=_require(env,'charging_slot_busy_minutes'); avail=_require(env,'charging_slot_available_minutes')
    return FormalRuntimeMetrics(
        len(delivered)/len(released) if released else 0.0, len(on_time)/len(released) if released else 0.0, len(on_time)/len(delivered) if delivered else 0.0, len(urgent_on_time)/len(urgent) if urgent else 0.0,
        fmean(lateness) if lateness else 0.0, max(lateness) if lateness else 0.0, len(released)-len(delivered), sum(float(getattr(t,'total_distance',0.0)) for t in _require(env,'trucks')),
        _ratio(_require(env,'truck_weight_utilization_sum'), dispatch), _ratio(_require(env,'truck_volume_utilization_sum'), dispatch), _ratio(_require(env,'truck_parcels_routed'), dispatch), float(_require(env,'bus_freight_utilization')),
        float(_require(env,'bus_propulsion_energy_kwh')), float(_require(env,'bus_charging_energy_kwh')), min(float(v) for v in bus_soc.values()), int(_require(env,'battery_safety_violation_count')),
        float(_require(env,'passenger_waiting_minutes')), float(_require(env,'passenger_onboard_delay_minutes')), float(_require(env,'raw_cost_components').get('bus_operating_delay',0.0)), int(_require(env,'drone_mission_count')),
        _ratio(busy, avail), float(_require(env,'locker_occupancy_kg_minutes')), float(_require(env,'peak_station_load_kw')), float(_require(env,'accumulated_power_overload')))

def collect_formal_metrics(env, *, env_reward=0.0, rlaif_rewards_by_agent=None, runtime=0.0):
    rlaif_rewards_by_agent=rlaif_rewards_by_agent or {}
    try:
        m=collect_formal_runtime_metrics(env)
        d=m.__dict__.copy()
    except FormalMetricError:
        # Backward-compatible lightweight collector for legacy tests/callers.
        parcels=list(getattr(env,'parcels',{}).values())
        released=[p for p in parcels if getattr(p,'release_time_min',None) is not None]
        delivered=[p for p in released if is_delivered_status(getattr(p,'status','')) and getattr(p,'delivered_time_min',None) is not None]
        urgent=[p for p in released if bool(getattr(p,'is_urgent', False))]
        on_time=[p for p in delivered if float(p.delivered_time_min)<=float(p.deadline_min)]
        urgent_on_time=[p for p in delivered if bool(getattr(p,'is_urgent', False)) and float(p.delivered_time_min)<=float(p.deadline_min)]
        lateness=[max(0.0,float(p.delivered_time_min)-float(p.deadline_min)) for p in delivered]
        d={k:0.0 for k in FORMAL_METRICS}
        d.update({'fulfillment_rate':len(delivered)/len(released) if released else 0.0,'on_time_over_all_released':len(on_time)/len(released) if released else 0.0,'on_time_over_delivered':len(on_time)/len(delivered) if delivered else 0.0,'urgent_on_time_fulfillment':len(urgent_on_time)/len(urgent) if urgent else 0.0,'average_lateness':fmean(lateness) if lateness else 0.0,'maximum_lateness':max(lateness) if lateness else 0.0,'undelivered_parcels':len(released)-len(delivered)})
    d['bus_energy']=d.get('bus_propulsion_energy_kwh',0.0)+d.get('bus_charging_energy_kwh',0.0)
    d.update({'total_normalized_cost':-float(env_reward),'environment_reward':float(env_reward),'rlaif_reward_assignment':float(rlaif_rewards_by_agent.get('assignment',0.0)),'rlaif_reward_bus':float(rlaif_rewards_by_agent.get('bus',0.0)),'rlaif_reward_truck':float(rlaif_rewards_by_agent.get('truck',0.0)),'rlaif_reward_station':float(rlaif_rewards_by_agent.get('station',0.0)),'runtime':float(runtime)})
    return d

def collect_environment_metrics(env, **kwargs):
    delivered=[p for p in env.parcels.values() if is_delivered_status(p.status) and p.delivered_time_min is not None]
    lateness=[max(0.0,float(p.delivered_time_min)-p.deadline_min) for p in delivered]
    return {'delivered_parcels':len(delivered),'undelivered_parcels':len(env.parcels)-len(delivered),'average_parcel_lateness':fmean(lateness) if lateness else 0.0, **env.get_metrics()}
