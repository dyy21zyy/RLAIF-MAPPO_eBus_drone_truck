"""Physical electric-bus circulation utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from envs.dynamics.passenger_dynamics import PassengerBusManifest
import math, random

EPS=1e-9

@dataclass
class RuntimePhysicalBus:
    physical_bus_id: str
    current_location: str
    soc_kwh: float
    battery_capacity_kwh: float = 160.0
    minimum_safe_energy_kwh: float = 40.0
    schedule_delay_min: float = 0.0
    next_available_time_min: float = 0.0
    current_trip_id: str | None = None
    onboard_parcel_ids: list[str] = field(default_factory=list)
    passenger_state: dict[str, Any] = field(default_factory=dict)
    passenger_manifest: PassengerBusManifest = field(default_factory=PassengerBusManifest)
    depleted: bool = False
    minimum_safe_energy_violation: bool = False
    relocation_status: str = "idle"
    last_relocation_energy_kwh: float = 0.0


def line_time_by_trip(stop_times: list[dict[str, Any]]) -> dict[str, float]:
    by: dict[str, list[dict[str, Any]]] = {}
    for row in stop_times:
        by.setdefault(str(row["trip_id"]), []).append(row)
    out = {}
    for tid, rows in by.items():
        rows.sort(key=lambda r: int(r.get("stop_sequence", 0)))
        out[tid] = float(rows[-1]["arrival_time"]) - float(rows[0]["departure_time"])
    return out


def calculate_physical_fleet_size(trips: list[dict[str, Any]], stop_times: list[dict[str, Any]], planned_headway_min: float, relocation_time_min: float, minimum_layover_min: float) -> dict[str, Any]:
    times = line_time_by_trip(stop_times)
    nominal_one_way = sum(times.values()) / max(len(times), 1)
    cycle = nominal_one_way + float(relocation_time_min) + float(minimum_layover_min)
    count = max(1, int(math.ceil(cycle / float(planned_headway_min))))
    return {"physical_bus_count": count, "nominal_one_way_line_time_min": nominal_one_way, "non_service_relocation_time_min": float(relocation_time_min), "minimum_layover_time_min": float(minimum_layover_min), "nominal_cycle_time_min": cycle, "planned_headway_min": float(planned_headway_min)}


def build_trip_to_bus_mapping(trips: list[dict[str, Any]], stop_times: list[dict[str, Any]], physical_bus_count: int, relocation_time_min: float, minimum_layover_min: float) -> list[dict[str, Any]]:
    line_times = line_time_by_trip(stop_times)
    ordered = sorted(trips, key=lambda r: (float(r.get("start_time", r.get("scheduled_start_min", 0))), str(r["trip_id"])))
    available = {f"bus_{i:03d}": 0.0 for i in range(physical_bus_count)}
    last_trip = {bid: "" for bid in available}
    seq = {bid: 0 for bid in available}
    rows=[]
    for trip in ordered:
        tid=str(trip["trip_id"]); start=float(trip.get("start_time", trip.get("scheduled_start_min", 0))); end=start+line_times[tid]
        candidates=sorted((t,b) for b,t in available.items() if t <= start + EPS)
        if not candidates:
            raise ValueError(f"No physical bus available for {tid}; overlapping assignment would be required")
        _, bid = candidates[0]
        rows.append({"trip_id":tid,"bus_id":bid,"sequence_index":seq[bid],"scheduled_start_min":round(start,6),"scheduled_end_min":round(end,6),"previous_trip_id":last_trip[bid],"next_trip_id":"","relocation_time_min":float(relocation_time_min),"minimum_layover_min":float(minimum_layover_min)})
        if last_trip[bid]:
            for r in reversed(rows):
                if r["trip_id"] == last_trip[bid]: r["next_trip_id"] = tid; break
        last_trip[bid]=tid; seq[bid]+=1; available[bid]=end+float(relocation_time_min)+float(minimum_layover_min)
    return rows


def sample_initial_energy(bus_ids: list[str], seed: int, capacity_kwh: float=160.0) -> dict[str,float]:
    rng=random.Random(int(seed))
    return {bid: rng.uniform(0.55*capacity_kwh, 0.85*capacity_kwh) for bid in sorted(bus_ids)}


def assert_no_overlaps(mapping: list[dict[str, Any]]) -> None:
    by={}
    for r in mapping: by.setdefault(r["bus_id"], []).append(r)
    for bid, rows in by.items():
        rows.sort(key=lambda r: float(r["scheduled_start_min"]))
        for a,b in zip(rows, rows[1:]):
            avail=float(a["scheduled_end_min"])+float(a["relocation_time_min"])+float(a["minimum_layover_min"])
            if avail > float(b["scheduled_start_min"])+EPS:
                raise ValueError(f"Overlapping physical-bus assignments for {bid}: {a['trip_id']} -> {b['trip_id']}")
