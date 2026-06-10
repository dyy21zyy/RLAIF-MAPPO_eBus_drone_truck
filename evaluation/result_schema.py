"""Stable per-episode result schema for Stage 8 experiments."""
from __future__ import annotations

RESULT_FIELDS = (
    "experiment_id", "method_name", "seed", "instance_name", "config_hash", "rlaif_enabled",
    "assignment_policy_name", "bus_policy_name", "episode_reward", "total_env_reward", "total_rlaif_reward",
    "delivered_parcels", "undelivered_parcels", "on_time_delivery_rate", "urgent_on_time_rate",
    "total_parcel_lateness", "average_parcel_lateness", "late_delivery_count", "truck_total_distance",
    "truck_operating_cost", "truck_direct_count", "truck_to_terminal_count", "truck_to_locker_count",
    "bus_charging_count", "bus_charging_energy", "passenger_delay", "bus_operating_delay", "minimum_bus_soc",
    "drone_delivery_count", "battery_shortage_count", "locker_overflow_amount", "locker_overflow_duration",
    "power_overload_amount", "power_overload_duration", "peak_station_load", "infeasible_action_count",
    "fallback_feasibility_events", "runtime_seconds", "status", "error_message",
)

NUMERIC_DEFAULTS = {field: 0.0 for field in RESULT_FIELDS if field not in {
    "experiment_id", "method_name", "instance_name", "config_hash", "assignment_policy_name", "bus_policy_name", "status", "error_message", "rlaif_enabled"
}}

def normalize_result(record):
    result = {field: NUMERIC_DEFAULTS.get(field, "") for field in RESULT_FIELDS}
    result.update(record)
    result["rlaif_enabled"] = bool(result["rlaif_enabled"])
    result["seed"] = int(result["seed"])
    return result
