"""Seeded synthetic parcel generation and drone-feasibility checks."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from data_pipeline.common import haversine_km, write_csv
from data_pipeline.download_osm import RoadGraph, nearest_node

PARCEL_COLUMNS = ["parcel_id", "release_time_min", "deadline_min", "deadline_class", "priority", "weight_kg", "volume_m3", "customer_lat", "customer_lon", "nearest_road_node", "reachable_station_ids", "nearest_reachable_station_id", "nearest_station_id", "drone_feasible", "nominal_feasible_modes", "release_time", "deadline", "deadline_type", "is_urgent", "weight", "volume"]


def calculate_drone_feasible(weight: float, station_lat: float, station_lon: float, customer_lat: float, customer_lon: float, config: dict[str, Any]) -> bool:
    network = config.get("network", {})
    drone = config.get("drone", {})
    one_way_km = haversine_km(station_lat, station_lon, customer_lat, customer_lon)
    speed = float(drone.get("speed_kmph", network.get("drone_speed_kmph", 40.0)))
    round_trip_min = 2 * one_way_km / speed * 60 + float(drone.get("customer_service_time_min", network.get("customer_service_time_min", 1.0)))
    return (weight <= float(drone.get("payload_capacity_kg", network.get("drone_payload_kg", 5.0)))
            and one_way_km <= float(drone.get("service_radius_one_way_km", network.get("drone_radius_km", 8.0)))
            and round_trip_min <= float(drone.get("maximum_round_trip_duration_min", network.get("max_drone_round_trip_min", 120.0))))


def _weighted_choice(rng: random.Random, choices: list[tuple[str, float]]) -> str:
    point = rng.random()
    cumulative = 0.0
    for value, probability in choices:
        cumulative += probability
        if point <= cumulative:
            return value
    return choices[-1][0]


def generate_parcels(config: dict[str, Any], graph: RoadGraph, stations: list[dict[str, Any]], output_dir: Path, trips: list[dict[str, Any]] | None = None, stop_times: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    from data_pipeline.deadline_generator import assign_deadline
    seed = int(config.get("seeds", {}).get("parcel_seed", config.get("project", {}).get("seed", 0)))
    rng = random.Random(seed)
    parcel_config = config.get("parcel", {})
    horizon = min(float(config.get("time", {}).get("bus_operation_horizon_min", config.get("bus", {}).get("horizon_min", 360.0))), float(config.get("time", {}).get("delivery_evaluation_horizon_min", config.get("bus", {}).get("delivery_horizon_min", config.get("bus", {}).get("horizon_min", 360.0)))))
    bbox = config.get("city", {}).get("bbox", {"south": 31.0, "north": 31.1, "west": 121.0, "east": 121.1})
    num = int(parcel_config.get("num_parcels", config.get("scenario", {}).get("num_parcels", 10)))
    formal_weights = parcel_config.get("weight_values_kg")
    fallback_only = bool(parcel_config.get("fallback_only", False)) or not bool(formal_weights)
    parcels = []
    for index in range(1, num + 1):
        release = rng.uniform(0, horizon)
        if formal_weights and not fallback_only:
            weight = float(rng.choice(formal_weights))
        else:
            weight_class = "heavy" if rng.random() < float(parcel_config.get("heavy_ratio", 0.1)) else rng.choice(["light", "medium"])
            weight_ranges = {"light": (0.5, 2.0), "medium": (2.0, 4.5), "heavy": (5.0, 15.0)}
            weight = rng.uniform(*weight_ranges[weight_class])
        max_retries = int(parcel_config.get("max_location_retries", 500))
        for attempt in range(max_retries):
            lat = rng.uniform(float(bbox["south"]), float(bbox["north"]))
            lon = rng.uniform(float(bbox["west"]), float(bbox["east"]))
            reachable = [str(st["station_id"]) for st in stations if calculate_drone_feasible(weight, float(st["lat"]), float(st["lon"]), lat, lon, config)]
            if reachable or fallback_only:
                break
        else:
            raise ValueError(f"Unable to generate drone-reachable customer for parcel_{index:04d} after {max_retries} retries")
        if not reachable:
            reachable = [str(min(stations, key=lambda item: haversine_km(lat, lon, float(item["lat"]), float(item["lon"])))["station_id"])]
        nearest = min((st for st in stations if str(st["station_id"]) in reachable), key=lambda item: haversine_km(lat, lon, float(item["lat"]), float(item["lon"])))
        parcel = {
            "parcel_id": f"parcel_{index:04d}", "release_time_min": round(release, 6),
            "weight_kg": round(weight, 6), "volume_m3": round(rng.uniform(float(parcel_config.get("volume_min_m3", 0.001)), float(parcel_config.get("volume_max_m3", 0.05))), 6),
            "customer_lat": round(lat, 7), "customer_lon": round(lon, 7),
            "nearest_road_node": nearest_node(graph, lat, lon), "reachable_station_ids": "|".join(reachable),
            "nearest_reachable_station_id": nearest["station_id"], "nearest_station_id": nearest["station_id"], "drone_feasible": True,
        }
        if trips is not None and stop_times is not None:
            parcel = assign_deadline(parcel, stations, trips, stop_times, config, rng)
        else:
            cls = _weighted_choice(rng, [("tight", .3), ("moderate", .5), ("loose", .2)])
            low, high = {"tight": (20, 40), "moderate": (40, 80), "loose": (80, 140)}[cls]
            parcel.update({"deadline_class": cls, "deadline_min": round(release + rng.uniform(low, high), 6), "priority": {"tight":3,"moderate":2,"loose":1}[cls], "nominal_feasible_modes": "TD"})
        parcel.update({"release_time": parcel["release_time_min"], "deadline": parcel["deadline_min"], "deadline_type": parcel["deadline_class"], "is_urgent": parcel["deadline_class"] == "tight", "weight": parcel["weight_kg"], "volume": parcel["volume_m3"]})
        if isinstance(parcel.get("nominal_feasible_modes"), list): parcel["nominal_feasible_modes"] = "|".join(parcel["nominal_feasible_modes"])
        parcels.append(parcel)
    write_csv(output_dir / "parcels.csv", parcels, PARCEL_COLUMNS)
    return parcels
