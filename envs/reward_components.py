"""Canonical reward-component registry for environment reward costs."""
from __future__ import annotations

REWARD_COMPONENTS = (
    "passenger_delay",
    "bus_operating_delay",
    "parcel_lateness",
    "energy_cost",
    "power_overload",
    "bus_battery_violation",
    "locker_overflow",
    "truck_cost",
    "undelivered",
    "battery_shortage",
    "infeasible_action",
)
REWARD_COMPONENT_SET = frozenset(REWARD_COMPONENTS)
RAW_COMPONENT_COLUMNS = tuple(f"raw_{name}" for name in REWARD_COMPONENTS)
