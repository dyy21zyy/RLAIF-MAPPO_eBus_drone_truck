"""Custom bus-route validation and deterministic fallback route generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data_pipeline.common import read_csv, write_csv
from data_pipeline.download_osm import RoadGraph

BUS_COLUMNS = ["route_id", "stop_id", "stop_name", "stop_sequence", "lat", "lon", "first_departure", "last_departure", "headway_min"]


def _fallback_stops(graph: RoadGraph, config: dict[str, Any]) -> list[dict[str, Any]]:
    ordered_nodes = sorted(graph.nodes, key=lambda node: (graph.nodes[node]["y"], graph.nodes[node]["x"]))
    stop_count = min(int(config.get("network", {}).get("num_stops", 20)), len(ordered_nodes))
    indices = [round(i * (len(ordered_nodes) - 1) / (stop_count - 1)) for i in range(stop_count)]
    bus = config["bus"]
    return [{
        "route_id": "fallback_corridor", "stop_id": f"stop_{sequence:02d}",
        "stop_name": f"Fallback Stop {sequence:02d}", "stop_sequence": sequence,
        "lat": graph.nodes[ordered_nodes[index]]["y"], "lon": graph.nodes[ordered_nodes[index]]["x"],
        "first_departure": "06:00", "last_departure": f"{6 + int(bus['horizon_min']) // 60:02d}:00",
        "headway_min": bus["headway_min"],
    } for sequence, index in enumerate(indices, start=1)]


def load_or_generate_bus_route(config: dict[str, Any], graph: RoadGraph, output_dir: Path, custom_csv: str | Path | None = None) -> list[dict[str, Any]]:
    candidate = Path(custom_csv) if custom_csv else Path(str(config.get("bus", {}).get("route_csv", "")))
    if str(candidate) not in ("", ".") and candidate.is_file():
        rows: list[dict[str, Any]] = read_csv(candidate)
        missing = set(BUS_COLUMNS) - set(rows[0] if rows else [])
        if missing:
            raise ValueError(f"Bus route CSV is missing columns: {sorted(missing)}")
        for row in rows:
            row["stop_sequence"] = int(row["stop_sequence"])
            row["lat"], row["lon"] = float(row["lat"]), float(row["lon"])
            row["headway_min"] = float(row["headway_min"])
        rows.sort(key=lambda row: row["stop_sequence"])
    else:
        rows = _fallback_stops(graph, config)
    write_csv(output_dir / "bus_stops.csv", rows, BUS_COLUMNS)
    return rows
