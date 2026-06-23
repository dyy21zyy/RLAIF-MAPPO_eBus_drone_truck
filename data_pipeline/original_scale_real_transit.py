"""Build original-scale instances using real transit inputs where available."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from data_pipeline.build_distance_matrices import build_distance_matrices
from data_pipeline.common import write_csv, write_json
from data_pipeline.download_osm import build_road_graph, save_road_graph
from data_pipeline.generate_depot import generate_depot
from data_pipeline.generate_integrated_stations import select_integrated_stations
from data_pipeline.generate_parcels import generate_parcels
from data_pipeline.original_ebus_drone import load_original_ebus_drone_defaults
from data_pipeline.real_transit import (
    load_simplified_stops_trips_csv,
    load_simplified_transit_csv,
    synthesize_stop_times_from_original_schedule,
)
from utils.config import PROJECT_ROOT, load_config

REAL_STOP_COLUMNS = [
    "route_id",
    "stop_id",
    "stop_name",
    "stop_sequence",
    "lat",
    "lon",
    "first_departure",
    "last_departure",
    "headway_min",
]
TRIP_COLUMNS = ["trip_id", "route_id", "direction_id", "service_id", "start_time", "freight_allowed"]
STOP_TIME_COLUMNS = ["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time", "freight_allowed"]


def _resolve_project_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _relative_or_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _filled(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def _derive_bbox_from_stops(stops: list[dict[str, Any]], margin: float = 0.01) -> dict[str, float]:
    lats = [float(row["lat"]) for row in stops]
    lons = [float(row["lon"]) for row in stops]
    return {
        "north": max(lats) + margin,
        "south": min(lats) - margin,
        "east": max(lons) + margin,
        "west": min(lons) - margin,
    }


def _select_real_stop_window(stops: list[dict[str, Any]], defaults: dict[str, Any], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep a real route window close to the inherited stop-count scale."""

    target = int(defaults["scale"]["num_stops"])
    tolerance = float(config["scale_policy"]["max_relative_deviation"]["num_stops"])
    warnings: list[str] = []
    ordered = sorted(stops, key=lambda row: (int(row["stop_sequence"]), row["stop_id"]))
    if len(ordered) > target * (1.0 + tolerance):
        selected = ordered[:target]
        warnings.append(
            f"Real route has {len(ordered)} stops; selected first {target} stops to match original eBus-Drone scale."
        )
    else:
        selected = ordered
        if len(selected) < target * (1.0 - tolerance):
            warnings.append(
                f"Real route has {len(selected)} stops, outside the configured tolerance from original {target} stops."
            )
    for sequence, row in enumerate(selected, start=1):
        row["stop_sequence"] = sequence
    return selected, warnings


def _normalise_runtime_config(config: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(config)
    scale = defaults["scale"]
    bus = defaults["bus"]
    charging = defaults["charging"]
    parcel = defaults["parcel"]
    drone = defaults["drone"]
    battery = defaults["battery"]
    power = defaults["power"]
    reward = defaults["reward"]

    cfg.setdefault("network", {})
    cfg["network"]["num_integrated_stations"] = int(
        _filled(cfg["network"].get("num_integrated_stations"), scale["num_integrated_stations"])
    )
    cfg["network"]["drone_radius_km"] = float(
        _filled(cfg["network"].get("drone_radius_km"), parcel["drone_service_radius_km"])
    )
    cfg["network"]["drone_speed_kmph"] = float(_filled(cfg["network"].get("drone_speed_kmph"), drone["speed_kmph"]))
    cfg["network"]["drone_payload_kg"] = float(_filled(cfg["network"].get("drone_payload_kg"), drone["payload_kg"]))
    cfg["network"]["max_drone_round_trip_min"] = float(
        _filled(cfg["network"].get("max_drone_round_trip_min"), drone["max_round_trip_duration_min"])
    )
    cfg["network"]["customer_service_time_min"] = float(
        _filled(cfg["network"].get("customer_service_time_min"), drone["customer_service_time_min"])
    )
    cfg["network"]["drone_turnaround_time_min"] = float(
        _filled(cfg["network"].get("drone_turnaround_time_min"), drone["turnaround_time_min"])
    )

    cfg.setdefault("bus", {})
    cfg["bus"]["horizon_min"] = float(_filled(cfg["bus"].get("horizon_min"), scale["bus_operation_horizon_min"]))
    cfg["bus"]["delivery_horizon_min"] = float(
        _filled(cfg["bus"].get("delivery_horizon_min"), scale["service_horizon_min"])
    )
    cfg["bus"]["headway_min"] = float(_filled(cfg["bus"].get("headway_min"), scale["planned_headway_min"]))
    cfg["bus"]["bus_speed_kmph"] = float(_filled(cfg["bus"].get("bus_speed_kmph"), bus["nominal_speed_kmph"]))
    cfg["bus"]["bus_capacity_passenger"] = int(_filled(cfg["bus"].get("bus_capacity_passenger"), bus["passenger_capacity"]))
    cfg["bus"]["bus_battery_kwh"] = float(_filled(cfg["bus"].get("bus_battery_kwh"), bus["battery_capacity_kwh"]))
    cfg["bus"]["bus_min_soc_kwh"] = float(_filled(cfg["bus"].get("bus_min_soc_kwh"), bus["safety_battery_kwh"]))
    cfg["bus"]["bus_energy_kwh_per_km"] = float(
        _filled(cfg["bus"].get("bus_energy_kwh_per_km"), bus["energy_kwh_per_km"])
    )
    cfg["bus"]["freight_capacity_kg"] = float(_filled(cfg["bus"].get("freight_capacity_kg"), bus["freight_capacity_kg"]))
    cfg["bus"]["terminal_loading_time_min_per_kg"] = float(
        _filled(cfg["bus"].get("terminal_loading_time_min_per_kg"), parcel["unloading_time_sec_per_kg"] / 60.0)
    )
    cfg["bus"]["station_unloading_time_min_per_kg"] = float(
        _filled(cfg["bus"].get("station_unloading_time_min_per_kg"), parcel["unloading_time_sec_per_kg"] / 60.0)
    )
    cfg["bus"]["charging_power_kw"] = float(_filled(cfg["bus"].get("charging_power_kw"), charging["pantograph_power_kw"]))
    cfg["bus"]["charging_efficiency"] = float(_filled(cfg["bus"].get("charging_efficiency"), charging["efficiency"]))
    cfg["bus"]["charging_actions_sec"] = list(
        _filled(cfg["bus"].get("charging_actions_sec"), charging["action_set_seconds"])
    )

    cfg.setdefault("station", {})
    cfg["station"]["chargers_per_station"] = int(
        _filled(cfg["station"].get("chargers_per_station"), charging["chargers_per_station"])
    )
    cfg["station"]["locker_capacity_kg"] = float(
        _filled(cfg["station"].get("locker_capacity_kg"), parcel["locker_capacity_kg"])
    )
    cfg["station"]["drones_per_station"] = int(_filled(cfg["station"].get("drones_per_station"), drone["drones_per_station"]))
    cfg["station"]["initial_full_batteries"] = int(
        _filled(cfg["station"].get("initial_full_batteries"), battery["initial_fully_charged_per_station"])
    )
    cfg["station"]["max_simultaneous_battery_charging"] = int(
        _filled(cfg["station"].get("max_simultaneous_battery_charging"), battery["max_simultaneous_charging"])
    )
    cfg["station"]["battery_charging_power_kw"] = float(
        _filled(cfg["station"].get("battery_charging_power_kw"), battery["charge_power_kw"])
    )
    cfg["station"]["battery_charging_duration_min"] = float(
        _filled(cfg["station"].get("battery_charging_duration_min"), battery["charge_duration_min"])
    )
    cfg["station"]["power_capacity_kw"] = float(_filled(cfg["station"].get("power_capacity_kw"), power["station_capacity_kw"]))
    cfg["station"]["base_load_kw"] = float(
        _filled(cfg["station"].get("base_load_kw"), (power["nominal_base_load_min_kw"] + power["nominal_base_load_max_kw"]) / 2)
    )

    cfg.setdefault("parcel", {})
    cfg["parcel"]["num_parcels"] = int(_filled(cfg["parcel"].get("num_parcels"), scale["num_parcels"]))
    deadline_mix = parcel["deadline_class_mix"]
    cfg["parcel"]["tight_deadline_ratio"] = float(
        _filled(cfg["parcel"].get("tight_deadline_ratio"), deadline_mix["tight"]["probability"])
    )
    cfg["parcel"]["moderate_deadline_ratio"] = float(
        _filled(cfg["parcel"].get("moderate_deadline_ratio"), deadline_mix["moderate"]["probability"])
    )
    cfg["parcel"]["loose_deadline_ratio"] = float(
        _filled(cfg["parcel"].get("loose_deadline_ratio"), deadline_mix["loose"]["probability"])
    )
    cfg["parcel"]["heavy_ratio"] = float(_filled(cfg["parcel"].get("heavy_ratio"), 0.0))

    cfg.setdefault("truck_extension", {})
    truck_defaults = {
        "num_trucks": 1,
        "capacity_kg": 100.0,
        "speed_kmph": bus["nominal_speed_kmph"],
        "cost_per_km": 2.0,
        "cost_per_min": 0.1,
        "loading_time_min_per_kg": 0.5,
        "unloading_time_min_per_kg": 0.4,
    }
    for key, value in truck_defaults.items():
        cfg["truck_extension"][key] = _filled(cfg["truck_extension"].get(key), value)
    cfg.setdefault("truck", {})
    for key, value in truck_defaults.items():
        cfg["truck"][key] = _filled(cfg["truck"].get(key), cfg["truck_extension"][key])
    cfg["truck"]["fixed_dispatch_cost"] = _filled(cfg["truck"].get("fixed_dispatch_cost"), 20.0)
    cfg["truck"]["return_to_depot"] = _filled(cfg["truck"].get("return_to_depot"), True)
    cfg["truck"]["one_parcel_per_trip_mvp"] = _filled(cfg["truck"].get("one_parcel_per_trip_mvp"), True)

    cfg.setdefault("reward", {})
    cfg["reward"]["passenger_delay"] = float(_filled(cfg["reward"].get("passenger_delay"), reward["alpha_1"]))
    cfg["reward"]["bus_operating_delay"] = float(_filled(cfg["reward"].get("bus_operating_delay"), reward["alpha_2"]))
    cfg["reward"]["parcel_lateness"] = float(_filled(cfg["reward"].get("parcel_lateness"), reward["alpha_4"]))
    cfg["reward"]["energy_cost"] = float(_filled(cfg["reward"].get("energy_cost"), reward["alpha_3"]))
    cfg["reward"]["power_overload"] = float(_filled(cfg["reward"].get("power_overload"), reward["alpha_6"]))
    cfg["reward"]["bus_battery_violation"] = float(_filled(cfg["reward"].get("bus_battery_violation"), reward["alpha_5"]))
    cfg["reward"]["locker_overflow"] = float(_filled(cfg["reward"].get("locker_overflow"), reward["eta_l_term"]))
    cfg["reward"]["battery_shortage"] = float(_filled(cfg["reward"].get("battery_shortage"), reward["eta_u_term"]))
    return cfg


def _load_transit(config: dict[str, Any], defaults: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[str]]:
    transit_cfg = config["transit"]
    stops_path = _resolve_project_path(transit_cfg["stops_csv"])
    trips_path = _resolve_project_path(transit_cfg["trips_csv"])
    stop_times_path = _resolve_project_path(transit_cfg.get("stop_times_csv"))
    provenance: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        transit = load_simplified_transit_csv(
            stops_path,
            trips_path,
            stop_times_path,
            route_id=transit_cfg.get("route_id"),
            direction_id=transit_cfg.get("direction_id"),
            service_id=transit_cfg.get("service_id"),
        )
        provenance.extend(
            [
                {
                    "field": "bus stops",
                    "value": len(transit["stops"]),
                    "source_type": "real_transit_data",
                    "source_file": str(stops_path),
                    "notes": "Real stop_id, stop_name, coordinates, and stop_sequence loaded from simplified CSV.",
                },
                {
                    "field": "bus trips",
                    "value": len(transit["trips"]),
                    "source_type": "real_transit_data",
                    "source_file": str(trips_path),
                    "notes": "Real trip_id, route_id, direction_id, and service_id loaded where present.",
                },
                {
                    "field": "bus stop_times",
                    "value": len(transit["stop_times"]),
                    "source_type": "real_transit_data",
                    "source_file": str(stop_times_path),
                    "notes": "Planned bus arrival/departure times are loaded from real stop_times CSV.",
                },
            ]
        )
    except FileNotFoundError as exc:
        if "stop_times" not in str(exc) or not bool(transit_cfg.get("allow_synthetic_timetable_if_missing", False)):
            raise
        stops_only = load_simplified_stops_trips_csv(
            stops_path,
            trips_path,
            route_id=transit_cfg.get("route_id"),
            direction_id=transit_cfg.get("direction_id"),
            service_id=transit_cfg.get("service_id"),
        )
        transit = synthesize_stop_times_from_original_schedule(
            stops_only["stops"],
            defaults,
            route_id=transit_cfg.get("route_id"),
        )
        provenance.extend(
            [
                {
                    "field": "bus stops",
                    "value": len(transit["stops"]),
                    "source_type": "real_transit_data",
                    "source_file": str(stops_path),
                    "notes": "Real stop_id, stop_name, coordinates, and stop_sequence loaded from simplified CSV.",
                },
                {
                    "field": "bus trips",
                    "value": len(transit["trips"]),
                    "source_type": "original_ebus_drone",
                    "source_file": "../eBus-Drone/configs/instances/medium.yaml",
                    "notes": "real stop_times unavailable; synthesized using original eBus-Drone schedule setting",
                },
                {
                    "field": "bus stop_times",
                    "value": len(transit["stop_times"]),
                    "source_type": "original_ebus_drone",
                    "source_file": "../eBus-Drone/configs/default.yaml",
                    "notes": "real stop_times unavailable; synthesized using original eBus-Drone schedule setting",
                },
            ]
        )
        warnings.append("Real stop_times unavailable; synthesized timetable from original eBus-Drone settings.")
    return transit, provenance, warnings


def _write_transit(output_dir: Path, transit: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    trips_by_id = {row["trip_id"]: row for row in transit["trips"]}
    stops = [
        {key: row.get(key, "") for key in REAL_STOP_COLUMNS}
        for row in transit["stops"]
    ]
    trips = [
        {key: row.get(key, "") for key in TRIP_COLUMNS}
        for row in transit["trips"]
    ]
    stop_times = []
    for row in transit["stop_times"]:
        trip = trips_by_id[row["trip_id"]]
        stop_times.append(
            {
                "trip_id": row["trip_id"],
                "stop_id": row["stop_id"],
                "stop_sequence": int(row["stop_sequence"]),
                "arrival_time": float(row["arrival_time"]),
                "departure_time": float(row["departure_time"]),
                "freight_allowed": trip.get("freight_allowed", "true"),
            }
        )
    write_csv(output_dir / "bus_stops.csv", stops, REAL_STOP_COLUMNS)
    write_csv(output_dir / "bus_trips.csv", trips, TRIP_COLUMNS)
    write_csv(output_dir / "bus_stop_times.csv", stop_times, STOP_TIME_COLUMNS)
    timetable = {
        "route_id": trips[0]["route_id"] if trips else "",
        "time_unit": "minutes_from_service_start",
        "source_policy": "real_transit_where_available_original_ebus_drone_where_missing",
        "trips": trips,
        "stop_times": stop_times,
    }
    write_json(output_dir / "bus_timetable.json", timetable)
    return trips, stop_times, timetable


def _scale_match_report(
    defaults: dict[str, Any],
    config: dict[str, Any],
    *,
    new_num_stops: int,
    new_num_integrated_stations: int,
    new_num_parcels: int,
    warnings: list[str],
) -> dict[str, Any]:
    scale = defaults["scale"]
    max_dev = config["scale_policy"]["max_relative_deviation"]
    comparisons = {
        "num_stops": (int(scale["num_stops"]), int(new_num_stops)),
        "num_integrated_stations": (int(scale["num_integrated_stations"]), int(new_num_integrated_stations)),
        "num_parcels": (int(scale["num_parcels"]), int(new_num_parcels)),
        "service_horizon_min": (float(scale["service_horizon_min"]), float(config["bus"]["delivery_horizon_min"])),
    }
    pass_flags = []
    for key, (original, new) in comparisons.items():
        allowed = float(max_dev.get(key, 0.25))
        relative = abs(float(new) - float(original)) / max(abs(float(original)), 1.0)
        pass_flags.append(relative <= allowed)
        if relative > allowed:
            warnings.append(f"{key} relative deviation {relative:.3f} exceeds tolerance {allowed:.3f}")
    return {
        "original_num_stops": comparisons["num_stops"][0],
        "new_num_stops": comparisons["num_stops"][1],
        "original_num_integrated_stations": comparisons["num_integrated_stations"][0],
        "new_num_integrated_stations": comparisons["num_integrated_stations"][1],
        "original_num_parcels": comparisons["num_parcels"][0],
        "new_num_parcels": comparisons["num_parcels"][1],
        "original_service_horizon_min": comparisons["service_horizon_min"][0],
        "new_service_horizon_min": comparisons["service_horizon_min"][1],
        "original_drone_radius_km": float(defaults["parcel"]["drone_service_radius_km"]),
        "new_drone_radius_km": float(config["network"]["drone_radius_km"]),
        "original_drone_payload_kg": float(defaults["drone"]["payload_kg"]),
        "new_drone_payload_kg": float(config["network"]["drone_payload_kg"]),
        "original_bus_charging_actions": list(defaults["charging"]["action_set_seconds"]),
        "new_bus_charging_actions": list(config["bus"]["charging_actions_sec"]),
        "scale_match_pass": all(pass_flags),
        "warnings": sorted(set(warnings)),
    }


def _provenance_entries(
    defaults: dict[str, Any],
    transit_entries: list[dict[str, Any]],
    config: dict[str, Any],
    stations: list[dict[str, Any]],
    parcels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        *transit_entries,
        {
            "field": "integrated stations",
            "value": len(stations),
            "source_type": "original_ebus_drone",
            "source_file": "../eBus-Drone/configs/instances/medium.yaml",
            "notes": "Selected from real bus stops using the original medium-scale station count and spacing logic.",
        },
        {
            "field": "parcels",
            "value": len(parcels),
            "source_type": "original_ebus_drone",
            "source_file": "../eBus-Drone/configs/instances/medium.yaml",
            "notes": "Generated using inherited parcel demand scale and deadline mix; not real parcel demand.",
        },
        {
            "field": "drone parameters",
            "value": {
                "speed_kmph": config["network"]["drone_speed_kmph"],
                "payload_kg": config["network"]["drone_payload_kg"],
                "radius_km": config["network"]["drone_radius_km"],
            },
            "source_type": "original_ebus_drone",
            "source_file": "../eBus-Drone/configs/default.yaml",
            "notes": "Inherited from previous eBus-Drone setting.",
        },
        {
            "field": "bus parameters",
            "value": {
                "speed_kmph": config["bus"]["bus_speed_kmph"],
                "freight_capacity_kg": config["bus"]["freight_capacity_kg"],
                "charging_actions_sec": config["bus"]["charging_actions_sec"],
            },
            "source_type": "original_ebus_drone",
            "source_file": "../eBus-Drone/configs/default.yaml",
            "notes": "Inherited bus system parameters; real stop_times override headway movement when available.",
        },
        {
            "field": "station parameters",
            "value": {
                "chargers_per_station": config["station"]["chargers_per_station"],
                "drones_per_station": config["station"]["drones_per_station"],
            },
            "source_type": "original_ebus_drone",
            "source_file": "../eBus-Drone/configs/default.yaml",
            "notes": "Inherited station equipment settings.",
        },
        {
            "field": "locker parameters",
            "value": {"locker_capacity_kg": config["station"]["locker_capacity_kg"]},
            "source_type": "original_ebus_drone",
            "source_file": "../eBus-Drone/configs/default.yaml",
            "notes": "Inherited locker capacity.",
        },
        {
            "field": "power parameters",
            "value": {
                "power_capacity_kw": config["station"]["power_capacity_kw"],
                "base_load_kw": config["station"]["base_load_kw"],
            },
            "source_type": "original_ebus_drone",
            "source_file": "../eBus-Drone/configs/default.yaml",
            "notes": "Inherited station power capacity and nominal base-load average.",
        },
        {
            "field": "reward weights",
            "value": config["reward"],
            "source_type": "original_ebus_drone",
            "source_file": "../eBus-Drone/configs/default.yaml",
            "notes": "Mapped from original reward alpha/eta coefficients plus explicit truck/RLAIF extension weights.",
        },
        {
            "field": "truck parameters",
            "value": config["truck_extension"],
            "source_type": "explicit_new_truck_extension",
            "source_file": "configs/original_scale_real_transit.yaml",
            "notes": "Truck is the new feeder layer; defaults are explicit assumptions calibrated not to dominate scale.",
        },
    ]


def build_original_scale_real_transit_instance(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a Stage 2-compatible instance for ``original_scale_real_transit``."""

    config = load_config(config_path)
    defaults = load_original_ebus_drone_defaults(config["reference"]["defaults_config"])
    config = _normalise_runtime_config(config, defaults)
    root = Path(output_root) if output_root else PROJECT_ROOT / "data" / "processed"
    output_dir = root / str(config["city"]["name"])
    output_dir.mkdir(parents=True, exist_ok=True)

    transit, transit_provenance, warnings = _load_transit(config, defaults)
    selected_stops, window_warnings = _select_real_stop_window(transit["stops"], defaults, config)
    selected_stop_ids = {row["stop_id"] for row in selected_stops}
    transit["stops"] = selected_stops
    transit["stop_times"] = [row for row in transit["stop_times"] if row["stop_id"] in selected_stop_ids]
    warnings.extend(window_warnings)

    bbox = config.get("city", {}).get("bbox", {})
    if not bbox or any(bbox.get(key) is None for key in ("north", "south", "east", "west")):
        config["city"]["bbox"] = _derive_bbox_from_stops(selected_stops)

    graph, graph_warnings = build_road_graph(config, fallback=not bool(config.get("network", {}).get("use_osm", False)))
    warnings.extend(graph_warnings)
    road_paths = save_road_graph(graph, output_dir)
    trips, stop_times, _timetable = _write_transit(output_dir, transit)
    stations = select_integrated_stations(selected_stops, config, output_dir)
    depot = generate_depot(config, graph, selected_stops, output_dir)
    parcels = generate_parcels(config, graph, stations, output_dir)
    matrix_paths, matrix_metadata, matrix_warnings = build_distance_matrices(
        graph, depot, selected_stops, stations, parcels, config, output_dir
    )
    warnings.extend(matrix_warnings)

    provenance = {
        "schema_version": 1,
        "data_mode": "original_scale_real_transit",
        "policy": "real transit where available; original eBus-Drone settings where real fields are unavailable; explicit truck extension.",
        "entries": _provenance_entries(defaults, transit_provenance, config, stations, parcels),
    }
    scale_report = _scale_match_report(
        defaults,
        config,
        new_num_stops=len(selected_stops),
        new_num_integrated_stations=len(stations),
        new_num_parcels=len(parcels),
        warnings=warnings,
    )
    write_json(output_dir / "data_provenance.json", provenance)
    write_json(output_dir / "scale_match_report.json", scale_report)

    artifact_names = {
        **{key: path.name for key, path in road_paths.items()},
        "depot": "depot.csv",
        "bus_stops": "bus_stops.csv",
        "bus_trips": "bus_trips.csv",
        "bus_stop_times": "bus_stop_times.csv",
        "bus_timetable": "bus_timetable.json",
        "integrated_stations": "integrated_stations.csv",
        "parcels": "parcels.csv",
        "data_provenance": "data_provenance.json",
        "scale_match_report": "scale_match_report.json",
        **{key: path.name for key, path in matrix_paths.items()},
    }
    instance = {
        "schema_version": 1,
        "stage": 2,
        "city_name": config["city"]["name"],
        "mode": "original_scale_real_transit",
        "data_mode": "original_scale_real_transit",
        "seed": config["project"]["seed"],
        "output_directory": str(output_dir),
        "artifacts": artifact_names,
        "counts": {
            "road_nodes": len(graph.nodes),
            "road_edges": len(graph.edges),
            "bus_stops": len(selected_stops),
            "bus_trips": len(trips),
            "bus_stop_times": len(stop_times),
            "integrated_stations": len(stations),
            "parcels": len(parcels),
        },
        "matrix_indices": matrix_metadata,
        "warnings": sorted(set(warnings)),
        "config_snapshot": config,
        "data_provenance": {
            entry["field"]: {
                "source_type": entry["source_type"],
                "source_file": entry["source_file"],
                "notes": entry["notes"],
                "value": _relative_or_value(entry["value"]),
            }
            for entry in provenance["entries"]
        },
        "scale_match_report": scale_report,
    }
    write_json(output_dir / "instance.json", instance)
    write_json(output_dir / "instance.yaml", instance)
    return instance
