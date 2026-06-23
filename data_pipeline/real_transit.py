"""Simplified real-transit CSV loader and inherited timetable synthesis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from data_pipeline.common import haversine_km, minutes_from_time, read_csv

STOP_COLUMNS = {"stop_id", "stop_name", "lat", "lon", "stop_sequence"}
TRIP_COLUMNS = {"trip_id", "route_id"}
STOP_TIME_COLUMNS = {"trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"}


def _require_columns(rows: list[dict[str, str]], required: set[str], path: Path) -> None:
    if not rows:
        raise ValueError(f"{path} is empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def _as_bool_text(value: Any, default: bool = True) -> str:
    if value is None or value == "":
        return str(default)
    return str(str(value).strip().lower() in {"true", "1", "yes", "y"}).lower()


def _time_min(value: str | int | float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(minutes_from_time(str(value)))


def load_simplified_transit_csv(
    stops_csv: str | Path,
    trips_csv: str | Path,
    stop_times_csv: str | Path | None,
    *,
    route_id: str | None = None,
    direction_id: str | int | None = None,
    service_id: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load real stop, trip, and stop-time CSV inputs without web access."""

    stops_path = Path(stops_csv)
    trips_path = Path(trips_csv)
    stop_times_path = Path(stop_times_csv) if stop_times_csv else None
    if not stops_path.is_file():
        raise FileNotFoundError(f"real bus stops CSV not found: {stops_path}")
    if not trips_path.is_file():
        raise FileNotFoundError(f"real bus trips CSV not found: {trips_path}")
    if stop_times_path is None or not stop_times_path.is_file():
        raise FileNotFoundError(f"real bus stop_times CSV not found: {stop_times_path}")

    raw_stops = read_csv(stops_path)
    raw_trips = read_csv(trips_path)
    raw_stop_times = read_csv(stop_times_path)
    _require_columns(raw_stops, STOP_COLUMNS, stops_path)
    _require_columns(raw_trips, TRIP_COLUMNS, trips_path)
    _require_columns(raw_stop_times, STOP_TIME_COLUMNS, stop_times_path)

    stops = [
        {
            "route_id": route_id or row.get("route_id") or raw_trips[0]["route_id"],
            "stop_id": str(row["stop_id"]),
            "stop_name": row["stop_name"],
            "stop_sequence": int(row["stop_sequence"]),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "first_departure": row.get("first_departure", "00:00"),
            "last_departure": row.get("last_departure", "23:59"),
            "headway_min": float(row.get("headway_min", 0) or 0),
            "source_type": "real_transit_data",
            "source_file": str(stops_path),
        }
        for row in raw_stops
    ]
    stops.sort(key=lambda row: (row["stop_sequence"], row["stop_id"]))
    stop_ids = {row["stop_id"] for row in stops}

    trips: list[dict[str, Any]] = []
    for row in raw_trips:
        if route_id is not None and str(row.get("route_id")) != str(route_id):
            continue
        if direction_id is not None and str(row.get("direction_id", "")) != str(direction_id):
            continue
        if service_id is not None and str(row.get("service_id", "")) != str(service_id):
            continue
        trips.append(
            {
                "trip_id": str(row["trip_id"]),
                "route_id": str(row["route_id"]),
                "direction_id": str(row.get("direction_id", "")),
                "service_id": str(row.get("service_id", "")),
                "start_time": 0.0,
                "freight_allowed": _as_bool_text(row.get("freight_allowed"), default=True),
                "source_type": "real_transit_data",
                "source_file": str(trips_path),
            }
        )
    if not trips:
        raise ValueError("No real trips remain after route/direction/service filtering")
    trip_ids = {row["trip_id"] for row in trips}

    stop_times: list[dict[str, Any]] = []
    for row in raw_stop_times:
        if str(row["trip_id"]) not in trip_ids or str(row["stop_id"]) not in stop_ids:
            continue
        stop_times.append(
            {
                "trip_id": str(row["trip_id"]),
                "stop_id": str(row["stop_id"]),
                "stop_sequence": int(row["stop_sequence"]),
                "arrival_time": round(_time_min(row["arrival_time"]), 6),
                "departure_time": round(_time_min(row["departure_time"]), 6),
                "source_type": "real_transit_data",
                "source_file": str(stop_times_path),
            }
        )
    if not stop_times:
        raise ValueError("No stop_times remain after trip and stop filtering")
    stop_times.sort(key=lambda row: (row["trip_id"], row["stop_sequence"]))
    first_by_trip: dict[str, float] = {}
    for row in stop_times:
        first_by_trip.setdefault(row["trip_id"], float(row["departure_time"]))
    for trip in trips:
        trip["start_time"] = first_by_trip.get(trip["trip_id"], 0.0)
    return {"stops": stops, "trips": trips, "stop_times": stop_times}


def load_simplified_stops_trips_csv(
    stops_csv: str | Path,
    trips_csv: str | Path,
    *,
    route_id: str | None = None,
    direction_id: str | int | None = None,
    service_id: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load real stops and trips when real stop_times are unavailable."""

    stops_path = Path(stops_csv)
    trips_path = Path(trips_csv)
    if not stops_path.is_file():
        raise FileNotFoundError(f"real bus stops CSV not found: {stops_path}")
    if not trips_path.is_file():
        raise FileNotFoundError(f"real bus trips CSV not found: {trips_path}")
    raw_stops = read_csv(stops_path)
    raw_trips = read_csv(trips_path)
    _require_columns(raw_stops, STOP_COLUMNS, stops_path)
    _require_columns(raw_trips, TRIP_COLUMNS, trips_path)

    trips: list[dict[str, Any]] = []
    for row in raw_trips:
        if route_id is not None and str(row.get("route_id")) != str(route_id):
            continue
        if direction_id is not None and str(row.get("direction_id", "")) != str(direction_id):
            continue
        if service_id is not None and str(row.get("service_id", "")) != str(service_id):
            continue
        trips.append(
            {
                "trip_id": str(row["trip_id"]),
                "route_id": str(row["route_id"]),
                "direction_id": str(row.get("direction_id", "")),
                "service_id": str(row.get("service_id", "")),
                "start_time": 0.0,
                "freight_allowed": _as_bool_text(row.get("freight_allowed"), default=True),
                "source_type": "real_transit_data",
                "source_file": str(trips_path),
            }
        )
    if not trips:
        raise ValueError("No real trips remain after route/direction/service filtering")
    route = route_id or trips[0]["route_id"]
    stops = [
        {
            "route_id": route,
            "stop_id": str(row["stop_id"]),
            "stop_name": row["stop_name"],
            "stop_sequence": int(row["stop_sequence"]),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "first_departure": row.get("first_departure", "00:00"),
            "last_departure": row.get("last_departure", "23:59"),
            "headway_min": float(row.get("headway_min", 0) or 0),
            "source_type": "real_transit_data",
            "source_file": str(stops_path),
        }
        for row in raw_stops
    ]
    stops.sort(key=lambda row: (row["stop_sequence"], row["stop_id"]))
    return {"stops": stops, "trips": trips}


def synthesize_stop_times_from_original_schedule(
    stops: list[dict[str, Any]],
    defaults: dict[str, Any],
    *,
    route_id: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Synthesize stop_times from original eBus-Drone schedule settings.

    This is allowed only when explicitly requested by config. The source type is
    ``original_ebus_drone`` because the timetable style is inherited, not real.
    """

    if not stops:
        raise ValueError("At least one real stop is required to synthesize stop_times")
    scale = defaults["scale"]
    bus = defaults["bus"]
    n_trips = int(scale["num_scheduled_bus_trips"])
    n_freight = int(scale["num_freight_carrying_trips"])
    headway = float(scale["planned_headway_min"])
    speed = float(bus["nominal_speed_kmph"])
    route = route_id or str(stops[0].get("route_id", "original_schedule"))
    trips: list[dict[str, Any]] = []
    stop_times: list[dict[str, Any]] = []
    freight_step = max(1, int(math.floor(n_trips / max(n_freight, 1))))
    for index in range(n_trips):
        trip_id = f"original_sched_{index + 1:03d}"
        start = index * headway
        freight_allowed = index % freight_step == 0 and sum(t["freight_allowed"] == "true" for t in trips) < n_freight
        trips.append(
            {
                "trip_id": trip_id,
                "route_id": route,
                "direction_id": "",
                "service_id": "original_ebus_drone_schedule",
                "start_time": start,
                "freight_allowed": str(bool(freight_allowed)).lower(),
                "source_type": "original_ebus_drone",
                "source_file": "../eBus-Drone/configs/instances/medium.yaml",
            }
        )
        clock = float(start)
        previous = None
        for stop in stops:
            if previous is not None:
                clock += haversine_km(
                    float(previous["lat"]),
                    float(previous["lon"]),
                    float(stop["lat"]),
                    float(stop["lon"]),
                ) / max(speed, 1e-9) * 60.0
            stop_times.append(
                {
                    "trip_id": trip_id,
                    "stop_id": stop["stop_id"],
                    "stop_sequence": int(stop["stop_sequence"]),
                    "arrival_time": round(clock, 6),
                    "departure_time": round(clock, 6),
                    "source_type": "original_ebus_drone",
                    "source_file": "../eBus-Drone/configs/default.yaml",
                }
            )
            previous = stop
    return {"stops": stops, "trips": trips, "stop_times": stop_times}
