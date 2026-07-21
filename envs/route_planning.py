"""Deterministic truck route planning helpers for batched truck routes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

STOP_CUSTOMER = "CUSTOMER"
STOP_BUS_TERMINAL = "BUS_TERMINAL"
STOP_INTEGRATED_STATION = "INTEGRATED_STATION"
STOP_DEPOT = "DEPOT"

@dataclass(frozen=True)
class RouteStop:
    stop_id: str
    stop_type: str
    parcel_ids: tuple[str, ...] = ()

@dataclass(frozen=True)
class RouteLeg:
    from_stop_id: str
    to_stop_id: str
    distance_km: float
    travel_time_min: float

@dataclass(frozen=True)
class RoutePlan:
    stops: tuple[RouteStop, ...]
    legs: tuple[RouteLeg, ...]
    total_distance_km: float
    total_travel_time_min: float


def destination_for(env: Any, parcel: Any) -> tuple[str, str]:
    mode = getattr(parcel, "mode", None)
    if mode == "TD":
        return parcel.parcel_id, STOP_CUSTOMER
    if mode == "TBD":
        terminal = None
        for rows in getattr(env, "trip_stop_times", {}).values():
            if rows:
                terminal = rows[0]["stop_id"]; break
        return terminal or "depot_01", STOP_BUS_TERMINAL
    if mode == "TLD":
        return str(parcel.station_id), STOP_INTEGRATED_STATION
    return parcel.parcel_id, STOP_CUSTOMER


def _dist_time(env: Any, a: str, b: str) -> tuple[float, float]:
    if a == b:
        return 0.0, 0.0
    ia = env.truck_location_index[a]; ib = env.truck_location_index[b]
    return float(env.truck_distance_m[ia, ib]) / 1000.0, float(env.truck_time_min[ia, ib])


def build_route(env: Any, start_location_id: str, parcels: Iterable[Any], return_to_depot: bool | None = None) -> RoutePlan:
    """Build a deterministic nearest-neighbor route over unique destinations."""
    groups: dict[tuple[str, str], list[str]] = {}
    for p in parcels:
        groups.setdefault(destination_for(env, p), []).append(p.parcel_id)
    remaining = [(sid, typ, tuple(sorted(ids))) for (sid, typ), ids in groups.items()]
    current = start_location_id
    ordered: list[RouteStop] = []
    legs: list[RouteLeg] = []
    total_d = total_t = 0.0
    while remaining:
        remaining.sort(key=lambda x: (_dist_time(env, current, x[0])[0], x[0], x[1]))
        sid, typ, ids = remaining.pop(0)
        d, t = _dist_time(env, current, sid)
        legs.append(RouteLeg(current, sid, d, t)); total_d += d; total_t += t
        ordered.append(RouteStop(sid, typ, ids)); current = sid
    if return_to_depot is None:
        return_to_depot = bool(env.config.get("truck", {}).get("return_to_depot", True))
    if return_to_depot and current != "depot_01":
        d, t = _dist_time(env, current, "depot_01")
        legs.append(RouteLeg(current, "depot_01", d, t)); total_d += d; total_t += t
        ordered.append(RouteStop("depot_01", STOP_DEPOT, ()))
    return RoutePlan(tuple(ordered), tuple(legs), total_d, total_t)
