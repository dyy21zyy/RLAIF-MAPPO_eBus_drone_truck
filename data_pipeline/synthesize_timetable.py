"""Generate deterministic bus trips and stop times."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data_pipeline.common import haversine_km, minutes_from_time, write_csv, write_json


def synthesize_timetable(stops: list[dict[str, Any]], config: dict[str, Any], output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not stops:
        raise ValueError("At least one bus stop is required")
    bus = config.get("bus", {})
    schedule = config.get("bus_schedule", {})
    horizon = float(config.get("time", {}).get("bus_operation_horizon_min", bus.get("horizon_min", 360.0)))
    if "scheduled_trip_count" in schedule:
        trip_count = int(schedule["scheduled_trip_count"])
        headway = float(schedule.get("planned_headway_min", horizon / max(trip_count, 1)))
        departures = [i * headway for i in range(trip_count)]
        departures = [d for d in departures if d < horizon - 1e-9]
        if len(departures) != trip_count:
            raise ValueError(f"Configured timetable creates {len(departures)} departures before {horizon}, expected {trip_count}")
        freight_count = int(schedule.get("freight_trip_count", 0))
        if freight_count > trip_count:
            raise ValueError("freight_trip_count cannot exceed scheduled_trip_count")
        freight_indices = set()
        if freight_count:
            for k in range(freight_count):
                freight_indices.add(round(k * (trip_count - 1) / max(freight_count - 1, 1)))
            i = 0
            while len(freight_indices) < freight_count:
                freight_indices.add(i); i += 1
        trips = [{"trip_id": f"trip_{index:03d}", "route_id": stops[0]["route_id"],
                  "start_time": departure, "freight_allowed": index in freight_indices}
                 for index, departure in enumerate(departures)]
    else:
        first = minutes_from_time(stops[0]["first_departure"])
        last = minutes_from_time(stops[0]["last_departure"])
        headway = float(stops[0].get("headway_min", bus["headway_min"]))
        departures = []
        value = first
        while value < last - 1e-9:
            departures.append(value)
            value += headway
        frequency = int(bus["freight_trip_frequency"])
        trips = [{"trip_id": f"trip_{index:03d}", "route_id": stops[0]["route_id"],
                  "start_time": departure, "freight_allowed": index % frequency == 0}
                 for index, departure in enumerate(departures)]
    stop_times: list[dict[str, Any]] = []
    speed = float(bus.get("bus_speed_kmph", bus.get("nominal_speed_kmph", 30.0)))
    for trip in trips:
        clock = float(trip["start_time"])
        previous = None
        for stop in stops:
            if previous is not None:
                clock += haversine_km(float(previous["lat"]), float(previous["lon"]), float(stop["lat"]), float(stop["lon"])) / speed * 60
            stop_times.append({"trip_id": trip["trip_id"], "stop_id": stop["stop_id"],
                               "stop_sequence": int(stop["stop_sequence"]), "arrival_time": round(clock, 6),
                               "departure_time": round(clock, 6), "freight_allowed": trip["freight_allowed"]})
            previous = stop
    # Phase 3 circulation derives nominal one-way line time from these stop times.
    timetable = {"route_id": stops[0]["route_id"], "time_unit": "minutes_after_midnight", "trips": trips, "stop_times": stop_times}
    write_csv(output_dir / "bus_trips.csv", trips, ["trip_id", "route_id", "start_time", "freight_allowed"])
    write_csv(output_dir / "bus_stop_times.csv", stop_times, ["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time", "freight_allowed"])
    write_json(output_dir / "bus_timetable.json", timetable)
    return trips, stop_times, timetable
