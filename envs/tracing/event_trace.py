from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any

@dataclass
class EventTraceRow:
    event_sequence: int
    scheduled_event_time: float
    actual_simulation_time: float
    event_kind: str
    event_type: str
    is_decision_event: bool
    active_agent: str | None = None
    entity_id: str | None = None
    trip_id: str | None = None
    physical_bus_id: str | None = None
    stop_id: str | None = None
    station_id: str | None = None
    parcel_id: str | None = None
    selected_action: int | None = None
    transition_id: str | None = None

class EventTraceCollector:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.rows: list[EventTraceRow] = []
    def append(self, row: EventTraceRow) -> None:
        if self.enabled:
            self.rows.append(row)
    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self.rows]
