from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any

@dataclass
class TruckTraceRow:
    event_sequence: int
    time_min: float
    truck_id: str
    event_kind: str
    route_index: int = 0
    stop_id: str | None = None
    parcel_ids: list[str] | None = None
    load_weight_kg: float = 0.0
    load_volume_m3: float = 0.0
    capacity_kg: float = 0.0
    capacity_m3: float = 0.0

class TruckTraceCollector:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled; self.rows: list[TruckTraceRow] = []
    def append(self, row: TruckTraceRow) -> None:
        if self.enabled: self.rows.append(row)
    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self.rows]
