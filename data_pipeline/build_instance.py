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
from data_pipeline.build_bus_circulation import build_bus_circulation
from data_pipeline.scenario_manifest import resolve_seeds, write_scenario_manifest
from utils.config import PROJECT_ROOT, load_config


def normalize_dynamic_config(config: dict[str, Any]) -> dict[str, Any]:
    """Map Phase-0 paper configuration names onto legacy pipeline keys."""
    config.setdefault("project", {}).setdefault("seed", 0)
    config.setdefault("city", {"name": f"shanghai_{config.get('scenario', {}).get('size', 'scenario')}", "center_lat": 31.23, "center_lon": 121.47, "bbox": {"south": 31.16, "north": 31.30, "west": 121.35, "east": 121.59}})
    if "bus_schedule" in config:
        config.setdefault("bus", {})["horizon_min"] = config.get("time", {}).get("bus_operation_horizon_min", 360)
        config["bus"].setdefault("headway_min", config["bus_schedule"].get("planned_headway_min", 10))
        config["bus"].setdefault("freight_trip_frequency", 3)
    bus = config.setdefault("bus", {})
    bus.setdefault("delivery_horizon_min", config.get("time", {}).get("delivery_evaluation_horizon_min", 480))
    bus.setdefault("bus_speed_kmph", bus.get("nominal_speed_kmph", 30.0))
    bus.setdefault("bus_battery_kwh", bus.get("battery_capacity_kwh", 160.0))
    bus.setdefault("bus_min_soc_kwh", bus.get("minimum_safe_energy_kwh", 40.0))
    bus.setdefault("bus_energy_kwh_per_km", bus.get("energy_consumption_kwh_per_km", 1.6))
    bus.setdefault("non_service_relocation_time_min", 5.0)
    bus.setdefault("minimum_layover_time_min", 2.0)
    config.setdefault("seeds", {}).setdefault("initial_bus_energy_seed", int(config.get("project", {}).get("seed", 0)) + 303)
    bus.setdefault("terminal_loading_time_min_per_kg", 0.0)
    bus.setdefault("station_unloading_time_min_per_kg", float(bus.get("unloading_time_sec_per_kg", 6.0)) / 60.0)
    net = config.setdefault("network", {})
    drone = config.get("drone", {})
    net.setdefault("drone_speed_kmph", drone.get("speed_kmph", 40.0))
    net.setdefault("drone_payload_kg", drone.get("payload_capacity_kg", 5.0))
    net.setdefault("drone_radius_km", drone.get("service_radius_one_way_km", 8.0))
    net.setdefault("max_drone_round_trip_min", drone.get("maximum_round_trip_duration_min", 120.0))
    net.setdefault("customer_service_time_min", drone.get("customer_service_time_min", 1.0))
    net.setdefault("drone_turnaround_time_min", drone.get("turnaround_time_min", 1.0))
    truck = config.setdefault("truck", {})
    truck.setdefault("capacity_kg", truck.get("weight_capacity_kg", 100.0))
    truck.setdefault("loading_time_min_per_kg", 0.0); truck.setdefault("unloading_time_min_per_kg", 0.0)
    truck.setdefault("fixed_dispatch_cost", 0.0); truck.setdefault("cost_per_min", 0.0); truck.setdefault("cost_per_km", 0.0)
    station = config.setdefault("station", {})
    station.setdefault("chargers_per_station", bus.get("chargers_per_station", 2))
    station.setdefault("drones_per_station", drone.get("drones_per_station", 3))
    station.setdefault("initial_full_batteries", config.get("drone_battery", {}).get("initial_full_batteries", 6))
    station.setdefault("battery_charging_power_kw", config.get("drone_battery", {}).get("charging_power_kw", 2.0))
    station.setdefault("battery_charging_duration_min", config.get("drone_battery", {}).get("charging_duration_min", 45.0))
    station.setdefault("base_load_kw", station.get("base_load_min_kw", 80.0))
    config.setdefault("reward", {k: 0.0 for k in ["passenger_delay","bus_operating_delay","parcel_lateness","energy_cost","power_overload","bus_battery_violation","locker_overflow","truck_cost","undelivered","battery_shortage","infeasible_action"]})
    return config

REQUIRED_FILENAMES = ["road_graph.graphml", "road_nodes.csv", "road_edges.csv", "depot.csv", "bus_stops.csv", "bus_trips.csv", "bus_stop_times.csv", "bus_timetable.json", "physical_buses.csv", "trip_to_bus.csv", "bus_circulation.json", "integrated_stations.csv", "parcels.csv", "truck_distance_matrix.npy", "truck_travel_time_matrix.npy", "drone_distance_matrix.npy", "instance.yaml", "instance.json"]


def build_instance(config_path: str | Path, fallback: bool = False, output_root: str | Path | None = None, custom_bus_route: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path)
    config = normalize_dynamic_config(load_config(config_path))
    config.setdefault("seeds", resolve_seeds(config))
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
    circulation = build_bus_circulation(trips, stop_times, config, output_dir)
    stations = select_integrated_stations(stops, config, output_dir)
    depot = generate_depot(config, graph, stops, output_dir)
    parcels = generate_parcels(config, graph, stations, output_dir, trips, stop_times)
    matrix_paths, matrix_metadata, matrix_warnings = build_distance_matrices(graph, depot, stops, stations, parcels, config, output_dir)
    warnings.extend(matrix_warnings)

    artifact_names = {**{key: path.name for key, path in road_paths.items()},
                      "depot": "depot.csv", "bus_stops": "bus_stops.csv", "bus_trips": "bus_trips.csv",
                      "bus_stop_times": "bus_stop_times.csv", "bus_timetable": "bus_timetable.json",
                      "physical_buses": "physical_buses.csv", "trip_to_bus": "trip_to_bus.csv", "bus_circulation": "bus_circulation.json",
                      "integrated_stations": "integrated_stations.csv", "parcels": "parcels.csv", "scenario_manifest": "scenario_manifest.json",
                      **{key: path.name for key, path in matrix_paths.items()}}
    instance = {"schema_version": 1, "stage": 2, "city_name": config["city"]["name"],
                "mode": "fallback" if fallback else ("fallback_after_osm_failure" if warnings else "full_osm"),
                "seed": config.get("project", {}).get("seed", 0), "seeds": config["seeds"], "output_directory": str(output_dir),
                "artifacts": artifact_names, "counts": {"road_nodes": len(graph.nodes), "road_edges": len(graph.edges),
                "bus_stops": len(stops), "bus_trips": len(trips), "bus_stop_times": len(stop_times),
                "physical_buses": len(circulation["physical_buses"]), "trip_to_bus": len(circulation["trip_to_bus"]),
                "integrated_stations": len(stations), "parcels": len(parcels)},
                "matrix_indices": matrix_metadata, "warnings": warnings, "config_snapshot": config}
    scenario_id = f"{config['city']['name']}-{config.get('scenario', {}).get('size', 'legacy')}"
    scenario_manifest = write_scenario_manifest(output_dir, scenario_id, config, {k:v for k,v in artifact_names.items() if k != "scenario_manifest"})
    instance["scenario_manifest"] = scenario_manifest
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
