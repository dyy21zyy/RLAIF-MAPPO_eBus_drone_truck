"""CLI for constructing the reproducible Stage 2 Shanghai instance."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from data_pipeline.build_distance_matrices import build_distance_matrices
from data_pipeline.common import write_json
from data_pipeline.download_osm import build_road_graph, save_road_graph
from data_pipeline.generate_depot import generate_depot
from data_pipeline.generate_integrated_stations import select_integrated_stations
from data_pipeline.generate_parcels import generate_parcels
from data_pipeline.parse_bus_route import load_or_generate_bus_route
from data_pipeline.synthesize_timetable import synthesize_timetable
from utils.config import PROJECT_ROOT, load_config

REQUIRED_FILENAMES = ["road_graph.graphml", "road_nodes.csv", "road_edges.csv", "depot.csv", "bus_stops.csv", "bus_trips.csv", "bus_stop_times.csv", "bus_timetable.json", "integrated_stations.csv", "parcels.csv", "truck_distance_matrix.npy", "truck_travel_time_matrix.npy", "drone_distance_matrix.npy", "instance.yaml", "instance.json"]


def build_instance(config_path: str | Path, fallback: bool = False, output_root: str | Path | None = None, custom_bus_route: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_config(config_path)
    if config.get("data_mode") == "original_scale_real_transit":
        from data_pipeline.original_scale_real_transit import build_original_scale_real_transit_instance

        if fallback:
            raise ValueError("original_scale_real_transit mode uses source-aware real/inherited data, not fallback=True")
        return build_original_scale_real_transit_instance(config_path, output_root=output_root)
    root = Path(output_root) if output_root else PROJECT_ROOT / "data" / "processed"
    output_dir = root / str(config["city"]["name"])
    output_dir.mkdir(parents=True, exist_ok=True)

    graph, warnings = build_road_graph(config, fallback=fallback)
    road_paths = save_road_graph(graph, output_dir)
    stops = load_or_generate_bus_route(config, graph, output_dir, custom_bus_route)
    trips, stop_times, _timetable = synthesize_timetable(stops, config, output_dir)
    stations = select_integrated_stations(stops, config, output_dir)
    depot = generate_depot(config, graph, stops, output_dir)
    parcels = generate_parcels(config, graph, stations, output_dir)
    matrix_paths, matrix_metadata, matrix_warnings = build_distance_matrices(graph, depot, stops, stations, parcels, config, output_dir)
    warnings.extend(matrix_warnings)

    artifact_names = {**{key: path.name for key, path in road_paths.items()},
                      "depot": "depot.csv", "bus_stops": "bus_stops.csv", "bus_trips": "bus_trips.csv",
                      "bus_stop_times": "bus_stop_times.csv", "bus_timetable": "bus_timetable.json",
                      "integrated_stations": "integrated_stations.csv", "parcels": "parcels.csv",
                      **{key: path.name for key, path in matrix_paths.items()}}
    instance = {"schema_version": 1, "stage": 2, "city_name": config["city"]["name"],
                "mode": "fallback" if fallback else ("fallback_after_osm_failure" if warnings else "full_osm"),
                "seed": config["project"]["seed"], "output_directory": str(output_dir),
                "artifacts": artifact_names, "counts": {"road_nodes": len(graph.nodes), "road_edges": len(graph.edges),
                "bus_stops": len(stops), "bus_trips": len(trips), "bus_stop_times": len(stop_times),
                "integrated_stations": len(stations), "parcels": len(parcels)},
                "matrix_indices": matrix_metadata, "warnings": warnings, "config_snapshot": config}
    write_json(output_dir / "instance.json", instance)
    # JSON is a strict subset of YAML, keeping fallback operation dependency-free.
    write_json(output_dir / "instance.yaml", instance)
    return instance


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fallback", action="store_true", help="Force the deterministic offline road graph and route")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--bus-route", type=Path, help="Optional custom route CSV")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    instance = build_instance(args.config, args.fallback, args.output_root, args.bus_route)
    print(f"Stage 2 instance built in {instance['output_directory']}")
    print(f"Mode: {instance['mode']}")
    for name in REQUIRED_FILENAMES:
        print(f"- {name}")
    for warning in instance["warnings"]:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
