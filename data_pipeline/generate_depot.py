"""Generate the single logistics depot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data_pipeline.common import write_csv
from data_pipeline.download_osm import RoadGraph, nearest_node


def generate_depot(config: dict[str, Any], graph: RoadGraph, stops: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    depot_config = config.get("depot", {})
    if "lat" in depot_config and "lon" in depot_config:
        lat, lon = float(depot_config["lat"]), float(depot_config["lon"])
        node = nearest_node(graph, lat, lon)
    else:
        terminal = min(stops, key=lambda stop: int(stop["stop_sequence"]))
        node = nearest_node(graph, float(terminal["lat"]), float(terminal["lon"]))
        lat, lon = graph.nodes[node]["y"], graph.nodes[node]["x"]
    depot = {"depot_id": "depot_01", "lat": lat, "lon": lon, "nearest_road_node": node}
    write_csv(output_dir / "depot.csv", [depot], ["depot_id", "lat", "lon", "nearest_road_node"])
    return depot
