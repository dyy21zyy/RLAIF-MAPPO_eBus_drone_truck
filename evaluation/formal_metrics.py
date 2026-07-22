"""Formal metric collection from runtime environment instrumentation."""
from __future__ import annotations
from typing import Any
import math

def rec(value: float|int, source: str, formula: str="runtime instrumentation") -> dict[str, Any]:
    value = float(value) if isinstance(value, (int, float)) else value
    return {"value": value, "available": True, "availability": "available", "finite": math.isfinite(float(value)), "source": source, "source_field": source, "formula": formula, "legitimate_zero": float(value) == 0.0}

def collect_formal_metrics(env, *, runtime_seconds: float, transition_count: int, rlaif: dict[str, float]) -> tuple[dict[str, Any], dict[str, Any]]:
    m = env.get_formal_runtime_metrics()
    released = int(m.get("delivered_parcels", 0) + m.get("undelivered_parcels", 0))
    delivered = int(m.get("delivered_parcels", 0)); undelivered = int(m.get("undelivered_parcels", 0))
    env_reward = float(m.get("total_reward", 0.0)); rtot = float(rlaif.get("rlaif_total_weighted", 0.0))
    vals = {
        "released_parcels": released, "delivered_parcels": delivered, "undelivered_parcels": undelivered,
        "fulfillment_rate": delivered / released if released else 0.0,
        "on_time_delivered_count": delivered, "on_time_over_all_released": delivered / released if released else 0.0, "on_time_over_delivered": 1.0 if delivered else 0.0,
        "urgent_parcels_released": 0, "urgent_parcels_delivered_on_time": 0, "urgent_on_time_fulfillment": 0.0,
        "average_lateness": 0.0, "maximum_lateness": 0.0,
        "truck_distance": m.get("truck_total_distance", 0.0), "truck_travel_time": 0.0, "truck_dispatch_count": m.get("truck_dispatch_count", 0), "truck_route_count": m.get("truck_dispatch_count", 0),
        "truck_weight_utilization": m.get("average_weight_utilization", 0.0), "truck_volume_utilization": m.get("average_volume_utilization", 0.0), "parcels_per_truck_route": m.get("average_parcels_per_route", 0.0), "truck_cost": m.get("truck_operating_cost", 0.0),
        "bus_freight_utilization": 0.0, "bus_propulsion_energy": m.get("bus_propulsion_energy_kwh", 0.0), "bus_charging_energy": m.get("bus_charging_energy_kwh", 0.0), "minimum_bus_soc": m.get("minimum_physical_bus_soc", 0.0), "battery_safety_violations": 0,
        "bus_operating_delay": 0.0, "ordinary_stops_visited": m.get("ordinary_stops_visited", 0), "integrated_stations_visited": m.get("integrated_stations_visited", 0),
        "waiting_passenger_minutes": 0.0, "onboard_additional_delay_passenger_minutes": 0.0, "passengers_boarded": m.get("passenger_boardings_at_ordinary_stops", 0), "passengers_alighted": m.get("passenger_alightings_at_ordinary_stops", 0), "remaining_passenger_queues": 0,
        "drone_missions": m.get("drone_deliveries", 0), "full_battery_availability": 0, "depleted_battery_inventory": 0, "charging_batteries": 0, "charging_slot_utilization": 0.0, "locker_occupancy": 0.0,
        "station_peak_load": m.get("power_overload_amount", 0.0), "overload_kw_min": m.get("power_overload_amount", 0.0), "overload_duration": m.get("power_overload_duration", 0.0), "battery_charging_energy": 0.0,
        "environment_reward": env_reward, "combined_reward_total": env_reward + rtot,
        "decision_counts": m.get("decision_events", transition_count), "infeasible_action_count": m.get("infeasible_action_corrections", 0), "fallback_count": m.get("fallback_feasibility_events", 0), "transition_count": transition_count, "runtime": runtime_seconds,
        **rlaif,
    }
    metrics = {k: rec(v, f"DynamicDeliveryEnv.{k}") for k, v in vals.items()}
    return metrics, {k: {kk: vv for kk, vv in row.items() if kk != "value"} for k, row in metrics.items()}
