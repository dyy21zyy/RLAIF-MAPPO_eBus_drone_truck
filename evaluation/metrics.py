"""Metric extraction from the Stage 3 environment without optional dependencies."""
from __future__ import annotations
from statistics import fmean


def collect_environment_metrics(env, *, bus_charging_count=0, bus_charging_energy=0.0, fallback_events=0):
    delivered = [p for p in env.parcels.values() if str(p.status).upper() == "DELIVERED" and p.delivered_time_min is not None]
    lateness = [max(0.0, float(p.delivered_time_min) - p.deadline_min) for p in delivered]
    urgent = [p for p in delivered if p.priority > 1]
    on_time = sum(value <= 0.0 for value in lateness)
    urgent_on_time = sum(float(p.delivered_time_min) <= p.deadline_min for p in urgent)
    station_count = len(env.station_ids)
    truck_distance = sum(float(truck.total_distance) for truck in env.trucks)
    costs = env.cost_components
    def raw_cost(name):
        weight = float(env.config["reward"].get(name, 1.0))
        return float(costs.get(name, 0.0)) / weight if weight else 0.0
    minimum_soc = min(env.bus_soc_kwh.values()) if env.bus_soc_kwh else 0.0
    modes = [p.mode for p in env.parcels.values()]
    return {
        "delivered_parcels": len(delivered), "undelivered_parcels": len(env.parcels)-len(delivered),
        "on_time_delivery_rate": on_time/len(delivered) if delivered else 0.0,
        "urgent_on_time_rate": urgent_on_time/len(urgent) if urgent else 0.0,
        "total_parcel_lateness": sum(lateness), "average_parcel_lateness": fmean(lateness) if lateness else 0.0,
        "late_delivery_count": sum(value > 0 for value in lateness), "truck_total_distance": truck_distance,
        "truck_operating_cost": raw_cost("truck_cost"), "truck_direct_count": modes.count("TD"),
        "truck_to_terminal_count": modes.count("TBD"), "truck_to_locker_count": modes.count("TLD"),
        "bus_charging_count": bus_charging_count, "bus_charging_energy": bus_charging_energy,
        "bus_loading_events": int(getattr(env, "decision_counts", {}).get("bus", 0)),
        "passenger_delay": raw_cost("passenger_delay"), "bus_operating_delay": raw_cost("bus_operating_delay"),
        "minimum_bus_soc": minimum_soc, "drone_delivery_count": sum(m in {"TBD", "TLD"} for m in modes),
        "battery_shortage_count": raw_cost("battery_shortage"),
        "locker_overflow_amount": float(getattr(env, "accumulated_locker_overflow", 0.0)),
        "locker_overflow_duration": float(getattr(env, "accumulated_locker_overflow_duration", 0.0)),
        "power_overload_amount": float(getattr(env, "accumulated_power_overload", 0.0)),
        "power_overload_duration": float(getattr(env, "accumulated_power_overload_duration", 0.0)),
        "peak_station_load": float(getattr(env, "peak_station_load_kw", 0.0)),
        "station_power_overload_kw_min": float(getattr(env, "accumulated_power_overload", 0.0)),
        "full_battery_count": float(sum(sum(b.status == "FULL" for b in getattr(st, "battery_states", [])) for st in getattr(env, "stations", {}).values())),
        "depleted_battery_count": float(sum(sum(b.status == "DEPLETED" for b in getattr(st, "battery_states", [])) for st in getattr(env, "stations", {}).values())),
        "charging_battery_count": float(sum(sum(b.status == "CHARGING" for b in getattr(st, "battery_states", [])) for st in getattr(env, "stations", {}).values())),
        "infeasible_action_count": env.infeasible_action_corrections, "fallback_feasibility_events": fallback_events,
    }

FORMAL_METRICS = (
    "total_normalized_cost", "environment_reward", "rlaif_reward_assignment", "rlaif_reward_bus", "rlaif_reward_truck", "rlaif_reward_station",
    "fulfillment_rate", "on_time_over_all_released", "on_time_over_delivered", "urgent_on_time_fulfillment", "average_lateness", "maximum_lateness",
    "undelivered_parcels", "truck_distance", "truck_weight_utilization", "truck_volume_utilization", "parcels_per_truck_route", "bus_freight_utilization",
    "bus_energy", "minimum_bus_soc", "battery_safety_violations", "waiting_passenger_minutes", "onboard_additional_delay_passenger_minutes", "bus_operating_delay",
    "drone_missions", "drone_utilization", "full_batteries", "depleted_batteries", "charging_batteries", "charging_slot_utilization", "locker_occupancy",
    "locker_overflow", "station_peak_power", "overload_kw_min", "infeasible_actions", "runtime"
)

def collect_formal_metrics(env, *, env_reward=0.0, rlaif_rewards_by_agent=None, runtime=0.0):
    """Collect paper metric names; urgent metrics use explicit parcel.is_urgent/urgent flags only."""
    rlaif_rewards_by_agent = rlaif_rewards_by_agent or {}
    parcels=list(getattr(env,'parcels',{}).values())
    released=[p for p in parcels if getattr(p,'release_time_min',0) is not None]
    delivered=[p for p in released if str(getattr(p,'status','')).upper()=='DELIVERED' and getattr(p,'delivered_time_min',None) is not None]
    urgent=[p for p in released if bool(getattr(p,'is_urgent',getattr(p,'urgent',False)))]
    on_time=[p for p in delivered if float(p.delivered_time_min)<=float(p.deadline_min)]
    urgent_on_time=[p for p in delivered if bool(getattr(p,'is_urgent',getattr(p,'urgent',False))) and float(p.delivered_time_min)<=float(p.deadline_min)]
    lateness=[max(0.0,float(p.delivered_time_min)-float(p.deadline_min)) for p in delivered]
    station_values=list(getattr(env,'stations',{}).values())
    formal={
        "total_normalized_cost": -float(env_reward), "environment_reward": float(env_reward),
        "rlaif_reward_assignment": float(rlaif_rewards_by_agent.get('assignment',0.0)), "rlaif_reward_bus": float(rlaif_rewards_by_agent.get('bus',0.0)),
        "rlaif_reward_truck": float(rlaif_rewards_by_agent.get('truck',0.0)), "rlaif_reward_station": float(rlaif_rewards_by_agent.get('station',0.0)),
        "fulfillment_rate": len(delivered)/len(released) if released else 0.0,
        "on_time_over_all_released": len(on_time)/len(released) if released else 0.0,
        "on_time_over_delivered": len(on_time)/len(delivered) if delivered else 0.0,
        "urgent_on_time_fulfillment": len(urgent_on_time)/len(urgent) if urgent else 0.0,
        "average_lateness": fmean(lateness) if lateness else 0.0, "maximum_lateness": max(lateness) if lateness else 0.0,
        "undelivered_parcels": len(released)-len(delivered), "truck_distance": sum(float(getattr(t,'total_distance',0.0)) for t in getattr(env,'trucks',[])),
        "truck_weight_utilization": float(getattr(env,'truck_weight_utilization',0.0)), "truck_volume_utilization": float(getattr(env,'truck_volume_utilization',0.0)),
        "parcels_per_truck_route": float(getattr(env,'parcels_per_truck_route',0.0)), "bus_freight_utilization": float(getattr(env,'bus_freight_utilization',0.0)),
        "bus_energy": float(getattr(env,'bus_energy_kwh',0.0)), "minimum_bus_soc": min(getattr(env,'bus_soc_kwh',{}).values()) if getattr(env,'bus_soc_kwh',{}) else 0.0,
        "battery_safety_violations": float(getattr(env,'battery_safety_violations',0.0)), "waiting_passenger_minutes": float(getattr(env,'waiting_passenger_minutes',0.0)),
        "onboard_additional_delay_passenger_minutes": float(getattr(env,'onboard_additional_delay_passenger_minutes',0.0)), "bus_operating_delay": float(getattr(env,'cost_components',{}).get('bus_operating_delay',0.0)),
        "drone_missions": float(getattr(env,'drone_missions',0.0)), "drone_utilization": float(getattr(env,'drone_utilization',0.0)),
        "full_batteries": float(sum(sum(getattr(b,'status','')=='FULL' for b in getattr(st,'battery_states',[])) for st in station_values)),
        "depleted_batteries": float(sum(sum(getattr(b,'status','')=='DEPLETED' for b in getattr(st,'battery_states',[])) for st in station_values)),
        "charging_batteries": float(sum(sum(getattr(b,'status','')=='CHARGING' for b in getattr(st,'battery_states',[])) for st in station_values)),
        "charging_slot_utilization": float(getattr(env,'charging_slot_utilization',0.0)), "locker_occupancy": float(getattr(env,'locker_occupancy',0.0)),
        "locker_overflow": float(getattr(env,'accumulated_locker_overflow',0.0)), "station_peak_power": float(getattr(env,'peak_station_load_kw',0.0)),
        "overload_kw_min": float(getattr(env,'accumulated_power_overload',0.0)), "infeasible_actions": float(getattr(env,'infeasible_action_corrections',0.0)), "runtime": float(runtime)
    }
    return formal
