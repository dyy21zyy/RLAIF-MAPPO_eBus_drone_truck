"""Event types for the final dynamic multi-agent contract.

Phase 0 only freezes names used by later phases; it does not implement event
execution semantics.
"""

from __future__ import annotations

from enum import Enum


class EventType(str, Enum):
    PARCEL_RELEASE = "PARCEL_RELEASE"
    TRUCK_AVAILABLE = "TRUCK_AVAILABLE"
    BUS_TRIP_DEPARTURE = "BUS_TRIP_DEPARTURE"
    BUS_TRIP_ARRIVAL = "BUS_TRIP_ARRIVAL"
    STATION_OPERATION = "STATION_OPERATION"
    DRONE_AVAILABLE = "DRONE_AVAILABLE"
    BATTERY_CHARGING_COMPLETE = "BATTERY_CHARGING_COMPLETE"
