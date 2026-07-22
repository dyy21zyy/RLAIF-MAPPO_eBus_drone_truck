"""Event types for the final dynamic multi-agent contract.

Phase 1 uses PARCEL_RELEASE as the sole transition from UNRELEASED to assignment-ready parcels; assignment decisions must not bind TBD parcels to scheduled trips or vehicles.
"""

from __future__ import annotations

from enum import Enum


class EventType(str, Enum):
    PARCEL_RELEASE = "PARCEL_RELEASE"
    TRUCK_AVAILABLE = "TRUCK_AVAILABLE"
    TRUCK_DEPARTURE = "TRUCK_DEPARTURE"
    TRUCK_ARRIVE_STOP = "TRUCK_ARRIVE_STOP"
    TRUCK_UNLOAD = "TRUCK_UNLOAD"
    TRUCK_ROUTE_COMPLETE = "TRUCK_ROUTE_COMPLETE"
    BUS_TRIP_DEPARTURE = "BUS_TRIP_DEPARTURE"
    BUS_TRIP_ARRIVAL = "BUS_TRIP_ARRIVAL"
    BUS_TRIP_START = "bus_trip_start"
    BUS_ARRIVE_STOP = "bus_arrive_stop"
    BUS_DEPART_STOP = "bus_depart_stop"
    BUS_TRIP_COMPLETE = "bus_trip_complete"
    BUS_TERMINAL_DEPARTURE = "BUS_TERMINAL_DEPARTURE"
    BUS_STATION_ARRIVAL = "BUS_STATION_ARRIVAL"
    PASSENGER_ARRIVAL = "PASSENGER_ARRIVAL"
    BUS_RELOCATION_COMPLETE = "BUS_RELOCATION_COMPLETE"
    STATION_OPERATION = "STATION_OPERATION"
    DRONE_AVAILABLE = "DRONE_AVAILABLE"
    BATTERY_CHARGING_COMPLETE = "BATTERY_CHARGING_COMPLETE"


# Canonical same-time event ordering. Lower numbers process first; the event
# heap then uses the stable sequence ID to make ties deterministic without
# mutating physical timestamps.
EVENT_PRIORITY = {
    "battery_ready": 0,
    "drone_return": 1,
    "station_operation": 2,
    "drone_dispatch": 3,
    "parcel_delivery": 4,
    "parcel_bus_terminal_arrival": 5,
    "parcel_station_arrival": 6,
    "truck_departure": 6,
    "truck_arrive_stop": 6,
    "truck_unload": 6,
    "truck_route_complete": 6,
    "bus_arrive_stop": 7,
    "bus_depart_stop": 8,
    "bus_trip_complete": 8,
    "BUS_RELOCATION_COMPLETE": 8,
    "bus_relocation_complete": 8,
    "bus_departure": 8,
    "truck_available": 9,
    "parcel_release": 10,
    "bus_trip_start": 11,
}
