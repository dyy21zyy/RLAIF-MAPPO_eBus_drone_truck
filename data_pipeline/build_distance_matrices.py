"""Build truck shortest-path and station-to-customer drone matrices."""

from __future__ import annotations

import heapq
import math
from pathlib import Path
from typing import Any

from data_pipeline.common import haversine_km, write_npy
from data_pipeline.download_osm import RoadGraph, nearest_node


def _shortest_paths(graph: RoadGraph, source: str, weight: str) -> dict[str, float]:
    adjacency: dict[str, list[tuple[str, float]]] = {node: [] for node in graph.nodes}
    for edge in graph.edges:
        adjacency.setdefault(str(edge["from_node"]), []).append((str(edge["to_node"]), float(edge[weight])))
    distances = {source: 0.0}
    queue = [(0.0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances.get(node):
            continue
        for target, cost in adjacency.get(node, []):
            candidate = distance + cost
            if candidate < distances.get(target, math.inf):
                distances[target] = candidate
                heapq.heappush(queue, (candidate, target))
    return distances


def build_distance_matrices(graph: RoadGraph, depot: dict[str, Any], stops: list[dict[str, Any]], stations: list[dict[str, Any]], parcels: list[dict[str, Any]], config: dict[str, Any], output_dir: Path) -> tuple[dict[str, Path], dict[str, Any], list[str]]:
    terminal = min(stops, key=lambda stop: int(stop["stop_sequence"]))
    locations = [{"id": depot["depot_id"], "type": "depot", "lat": float(depot["lat"]), "lon": float(depot["lon"]), "road_node": depot["nearest_road_node"]},
                 {"id": terminal["stop_id"], "type": "terminal", "lat": float(terminal["lat"]), "lon": float(terminal["lon"]), "road_node": nearest_node(graph, float(terminal["lat"]), float(terminal["lon"]))}]
    locations.extend({"id": item["station_id"], "type": "integrated_station", "lat": float(item["lat"]), "lon": float(item["lon"]), "road_node": nearest_node(graph, float(item["lat"]), float(item["lon"]))} for item in stations)
    locations.extend({"id": item["parcel_id"], "type": "customer", "lat": float(item["customer_lat"]), "lon": float(item["customer_lon"]), "road_node": item["nearest_road_node"]} for item in parcels)
    warnings: list[str] = []
    size = len(locations)
    distance_matrix = [[0.0] * size for _ in range(size)]
    time_matrix = [[0.0] * size for _ in range(size)]
    distance_cache = {node: _shortest_paths(graph, node, "length_m") for node in {item["road_node"] for item in locations}}
    time_cache = {node: _shortest_paths(graph, node, "travel_time_min") for node in {item["road_node"] for item in locations}}
    truck_speed = float(config["truck"]["speed_kmph"])
    for row, source in enumerate(locations):
        for column, target in enumerate(locations):
            distance_m = distance_cache[source["road_node"]].get(target["road_node"])
            travel_min = time_cache[source["road_node"]].get(target["road_node"])
            if distance_m is None or travel_min is None:
                distance_m = haversine_km(source["lat"], source["lon"], target["lat"], target["lon"]) * 1000
                travel_min = distance_m / 1000 / truck_speed * 60
                warnings.append(f"No road path from {source['id']} to {target['id']}; used haversine fallback")
            distance_matrix[row][column] = distance_m
            time_matrix[row][column] = travel_min
    drone_matrix = [[haversine_km(float(station["lat"]), float(station["lon"]), float(parcel["customer_lat"]), float(parcel["customer_lon"])) * 1000 for parcel in parcels] for station in stations]
    paths = {"truck_distance_matrix": output_dir / "truck_distance_matrix.npy",
             "truck_travel_time_matrix": output_dir / "truck_travel_time_matrix.npy",
             "drone_distance_matrix": output_dir / "drone_distance_matrix.npy"}
    write_npy(paths["truck_distance_matrix"], distance_matrix)
    write_npy(paths["truck_travel_time_matrix"], time_matrix)
    write_npy(paths["drone_distance_matrix"], drone_matrix)
    metadata = {"truck_locations": locations,
                "drone_rows": [{"index": i, "station_id": station["station_id"]} for i, station in enumerate(stations)],
                "drone_columns": [{"index": i, "parcel_id": parcel["parcel_id"]} for i, parcel in enumerate(parcels)],
                "units": {"truck_distance": "metres", "truck_travel_time": "minutes", "drone_distance": "metres"}}
    return paths, metadata, sorted(set(warnings))
