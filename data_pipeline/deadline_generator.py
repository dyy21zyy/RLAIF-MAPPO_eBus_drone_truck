"""Deadline and nominal feasibility helpers for dynamic parcel scenarios."""
from __future__ import annotations

import random
from typing import Any

from data_pipeline.common import haversine_km

DEADLINE_CLASS_PROBABILITIES = (("tight", 0.30), ("moderate", 0.50), ("loose", 0.20))
SLACK_RANGES_MIN = {"tight": (20.0, 40.0), "moderate": (40.0, 80.0), "loose": (80.0, 140.0)}


def choose_deadline_class(rng: random.Random) -> str:
    point = rng.random(); total = 0.0
    for name, prob in DEADLINE_CLASS_PROBABILITIES:
        total += prob
        if point <= total:
            return name
    return DEADLINE_CLASS_PROBABILITIES[-1][0]


def nominal_drone_minutes(distance_km: float, config: dict[str, Any]) -> float:
    drone = config.get("drone", {})
    network = config.get("network", {})
    speed = float(drone.get("speed_kmph", network.get("drone_speed_kmph", 40.0)))
    service = float(drone.get("customer_service_time_min", network.get("customer_service_time_min", 1.0)))
    return distance_km / max(speed, 1e-9) * 60.0 + service


def nominal_feasible_completions(parcel: dict[str, Any], stations: list[dict[str, Any]], trips: list[dict[str, Any]], stop_times: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    """Return nominal earliest completion times by mode without fixing a plan."""
    release = float(parcel["release_time_min"])
    weight = float(parcel["weight_kg"])
    truck = config.get("truck", {})
    drone_cfg = config.get("drone", {})
    time_cfg = config.get("time", {})
    t_del = float(time_cfg.get("delivery_evaluation_horizon_min", config.get("bus", {}).get("delivery_horizon_min", 480.0)))
    truck_speed = float(truck.get("speed_kmph", 25.0))
    service = float(drone_cfg.get("customer_service_time_min", config.get("network", {}).get("customer_service_time_min", 1.0)))
    depot_lat = float(config.get("city", {}).get("center_lat", stations[0]["lat"] if stations else parcel["customer_lat"]))
    depot_lon = float(config.get("city", {}).get("center_lon", stations[0]["lon"] if stations else parcel["customer_lon"]))
    cust_lat, cust_lon = float(parcel["customer_lat"]), float(parcel["customer_lon"])
    completions = {"TD": release + haversine_km(depot_lat, depot_lon, cust_lat, cust_lon) / truck_speed * 60.0 + service}
    reachable = set(parcel.get("reachable_station_ids", []))
    if isinstance(parcel.get("reachable_station_ids"), str):
        reachable = set(str(parcel["reachable_station_ids"]).split("|"))
    stop_by_station = {str(s.get("station_id")): str(s.get("stop_id", s.get("station_id"))) for s in stations}
    freight_ids = {str(t["trip_id"]) for t in trips if str(t.get("freight_allowed", "false")).lower() in {"true", "1", "yes"}}
    times_by_trip: dict[str, list[dict[str, Any]]] = {}
    for row in stop_times:
        times_by_trip.setdefault(str(row["trip_id"]), []).append(row)
    for rows in times_by_trip.values():
        rows.sort(key=lambda r: int(r.get("stop_sequence", 0)))
    for s in stations:
        sid = str(s["station_id"])
        if reachable and sid not in reachable:
            continue
        d = haversine_km(float(s["lat"]), float(s["lon"]), cust_lat, cust_lon)
        drone_oneway = nominal_drone_minutes(d, config)
        station_truck = haversine_km(depot_lat, depot_lon, float(s["lat"]), float(s["lon"])) / truck_speed * 60.0
        completions[f"TLD_{sid}"] = release + station_truck + drone_oneway
        for trip_id in freight_ids:
            rows = times_by_trip.get(trip_id, [])
            if not rows: continue
            dep = float(rows[0].get("departure_time", rows[0].get("start_time", 0.0)))
            target = next((r for r in rows if str(r.get("stop_id")) == stop_by_station.get(sid)), None)
            if target and dep >= release:
                unload = weight * float(config.get("bus", {}).get("unloading_time_sec_per_kg", 6.0)) / 60.0
                completions[f"TBD_{sid}"] = min(completions.get(f"TBD_{sid}", float("inf")), float(target["arrival_time"]) + unload + drone_oneway)
    return {k:v for k,v in completions.items() if v <= t_del}


def assign_deadline(parcel: dict[str, Any], stations: list[dict[str, Any]], trips: list[dict[str, Any]], stop_times: list[dict[str, Any]], config: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    completions = nominal_feasible_completions(parcel, stations, trips, stop_times, config)
    if not completions:
        raise ValueError(f"Parcel {parcel.get('parcel_id')} has no nominally feasible delivery mode before T_del")
    cls = choose_deadline_class(rng)
    lo, hi = SLACK_RANGES_MIN[cls]
    parcel["deadline_class"] = cls
    parcel["deadline_min"] = round(min(completions.values()) + rng.uniform(lo, hi), 6)
    parcel["priority"] = {"tight": 3, "moderate": 2, "loose": 1}[cls]
    parcel["nominal_feasible_modes"] = sorted(completions)
    return parcel
