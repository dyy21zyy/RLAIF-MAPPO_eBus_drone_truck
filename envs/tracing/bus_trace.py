from __future__ import annotations
from dataclasses import asdict, dataclass

@dataclass
class BusStopTraceRow:
    event_sequence: int
    physical_bus_id: str
    trip_id: str
    stop_index: int
    stop_id: str
    integrated_station: bool
    scheduled_arrival: float
    actual_arrival: float
    scheduled_departure: float
    actual_departure: float
    passenger_alighting: int = 0
    passenger_boarding: int = 0
    onboard_passengers_after_departure: int = 0
    freight_loaded: int = 0
    freight_unloaded: int = 0
    onboard_freight_after_departure: int = 0
    charging_duration_min: float = 0.0
    soc_before_incoming_segment: float = 0.0
    segment_energy_kwh: float = 0.0
    soc_at_arrival: float = 0.0
    charging_energy_kwh: float = 0.0
    soc_at_departure: float = 0.0
    current_trip_delay_min: float = 0.0
    cumulative_bus_delay_min: float = 0.0

class BusTraceCollector:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.rows: list[BusStopTraceRow] = []
    def append(self, row: BusStopTraceRow) -> None:
        if self.enabled:
            self.rows.append(row)
    def as_dicts(self) -> list[dict]:
        return [asdict(r) for r in self.rows]
