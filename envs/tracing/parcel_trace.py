from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any

@dataclass
class ParcelTraceRow:
    event_sequence: int
    time_min: float
    parcel_id: str
    status: str
    mode: str | None = None
    station_id: str | None = None
    truck_id: str | None = None
    event_kind: str | None = None

class ParcelTraceCollector:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled; self.rows: list[ParcelTraceRow] = []
    def append(self, row: ParcelTraceRow) -> None:
        if self.enabled: self.rows.append(row)
    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self.rows]
