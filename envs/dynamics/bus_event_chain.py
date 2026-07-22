"""Canonical stop-by-stop physical bus event chain primitives."""
from __future__ import annotations
from dataclasses import dataclass, field

BUS_TRIP_START = "bus_trip_start"
BUS_ARRIVE_STOP = "bus_arrive_stop"
BUS_DEPART_STOP = "bus_depart_stop"
BUS_TRIP_COMPLETE = "bus_trip_complete"
BUS_RELOCATION_COMPLETE = "bus_relocation_complete"
BUS_TERMINAL_DEPARTURE = "BUS_TERMINAL_DEPARTURE"
BUS_STATION_ARRIVAL = "BUS_STATION_ARRIVAL"

AUTOMATIC_BUS_EVENTS = frozenset({BUS_TRIP_START, BUS_ARRIVE_STOP, BUS_DEPART_STOP, BUS_TRIP_COMPLETE, BUS_RELOCATION_COMPLETE})
DECISION_BUS_EVENTS = frozenset({BUS_TERMINAL_DEPARTURE, BUS_STATION_ARRIVAL})

@dataclass
class RuntimeTripState:
    trip_id: str
    physical_bus_id: str
    scheduled_start_min: float
    actual_start_min: float | None = None
    current_stop_index: int = 0
    actual_arrival_times: dict[int, float] = field(default_factory=dict)
    actual_departure_times: dict[int, float] = field(default_factory=dict)
    visited_stop_indices: list[int] = field(default_factory=list)
    completed: bool = False
    completion_time_min: float | None = None
    freight_allowed: bool = False

@dataclass(frozen=True)
class BusSegment:
    trip_id: str
    from_stop_index: int
    to_stop_index: int
    from_stop_id: str
    to_stop_id: str
    scheduled_running_time_min: float
    distance_km: float
