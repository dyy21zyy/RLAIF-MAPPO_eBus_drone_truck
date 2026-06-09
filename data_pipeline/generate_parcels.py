"""Seeded synthetic parcel generation and drone-feasibility checks."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from data_pipeline.common import haversine_km, write_csv
from data_pipeline.download_osm import RoadGraph, nearest_node

PARCEL_COLUMNS = ["parcel_id", "release_time", "deadline", "deadline_type", "weight", "volume", "priority", "customer_lat", "customer_lon", "nearest_road_node", "nearest_station_id", "drone_feasible"]


def calculate_drone_feasible(weight: float, station_lat: float, station_lon: float, customer_lat: float, customer_lon: float, config: dict[str, Any]) -> bool:
    network = config["network"]
    round_trip_km = 2 * haversine_km(station_lat, station_lon, customer_lat, customer_lon)
    round_trip_min = round_trip_km / float(network["drone_speed_kmph"]) * 60
    return (weight <= float(network["drone_payload_kg"])
            and round_trip_km <= float(network["drone_radius_km"])
            and round_trip_min <= float(network["max_drone_round_trip_min"]))


def _weighted_choice(rng: random.Random, choices: list[tuple[str, float]]) -> str:
    point = rng.random()
    cumulative = 0.0
    for value, probability in choices:
        cumulative += probability
        if point <= cumulative:
            return value
    return choices[-1][0]


def generate_parcels(config: dict[str, Any], graph: RoadGraph, stations: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    rng = random.Random(int(config["project"]["seed"]))
    parcel_config = config["parcel"]
    horizon = float(config["bus"]["horizon_min"])
    bbox = config["city"]["bbox"]
    deadline_choices = [("tight", float(parcel_config["tight_deadline_ratio"])),
                        ("moderate", float(parcel_config["moderate_deadline_ratio"])),
                        ("loose", float(parcel_config["loose_deadline_ratio"]))]
    parcels = []
    for index in range(1, int(parcel_config["num_parcels"]) + 1):
        release = rng.uniform(0, horizon)
        deadline_type = _weighted_choice(rng, deadline_choices)
        deadline_ranges = {"tight": (20, 40), "moderate": (40, 80), "loose": (80, 140)}
        low, high = deadline_ranges[deadline_type]
        weight_class = "heavy" if rng.random() < float(parcel_config["heavy_ratio"]) else rng.choice(["light", "medium"])
        weight_ranges = {"light": (0.5, 2.0), "medium": (2.0, 4.5), "heavy": (5.0, 15.0)}
        weight = rng.uniform(*weight_ranges[weight_class])
        lat = rng.uniform(float(bbox["south"]), float(bbox["north"]))
        lon = rng.uniform(float(bbox["west"]), float(bbox["east"]))
        station = min(stations, key=lambda item: haversine_km(lat, lon, float(item["lat"]), float(item["lon"])))
        parcels.append({
            "parcel_id": f"parcel_{index:04d}", "release_time": round(release, 6),
            "deadline": round(release + rng.uniform(low, high), 6), "deadline_type": deadline_type,
            "weight": round(weight, 6), "volume": round(rng.uniform(0.001, 0.05), 6),
            "priority": {"tight": 3, "moderate": 2, "loose": 1}[deadline_type],
            "customer_lat": round(lat, 7), "customer_lon": round(lon, 7),
            "nearest_road_node": nearest_node(graph, lat, lon), "nearest_station_id": station["station_id"],
            "drone_feasible": calculate_drone_feasible(weight, float(station["lat"]), float(station["lon"]), lat, lon, config),
        })
    write_csv(output_dir / "parcels.csv", parcels, PARCEL_COLUMNS)
    return parcels
