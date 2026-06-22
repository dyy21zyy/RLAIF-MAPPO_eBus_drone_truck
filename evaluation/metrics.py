"""Metric extraction from the Stage 3 environment without optional dependencies."""
from __future__ import annotations
from statistics import fmean


def collect_environment_metrics(env, *, bus_charging_count=0, bus_charging_energy=0.0, fallback_events=0):
    delivered = [p for p in env.parcels.values() if p.status == "delivered" and p.delivered_time_min is not None]
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
        "passenger_delay": raw_cost("passenger_delay"), "bus_operating_delay": raw_cost("bus_operating_delay"),
        "minimum_bus_soc": minimum_soc, "drone_delivery_count": sum(m in {"TBD", "TLD"} for m in modes),
        "battery_shortage_count": raw_cost("battery_shortage"),
        "locker_overflow_amount": float(getattr(env, "accumulated_locker_overflow", 0.0)),
        "locker_overflow_duration": float(getattr(env, "accumulated_locker_overflow_duration", 0.0)),
        "power_overload_amount": float(getattr(env, "accumulated_power_overload", 0.0)),
        "power_overload_duration": float(getattr(env, "accumulated_power_overload_duration", 0.0)),
        "peak_station_load": float(getattr(env, "peak_station_load_kw", 0.0)),
        "infeasible_action_count": env.infeasible_action_corrections, "fallback_feasibility_events": fallback_events,
    }
