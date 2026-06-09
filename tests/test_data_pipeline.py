"""Tests for the dependency-light Stage 2 Shanghai data pipeline."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from data_pipeline.build_instance import REQUIRED_FILENAMES, build_instance
from data_pipeline.common import read_csv
from data_pipeline.download_osm import create_fallback_graph
from data_pipeline.generate_integrated_stations import select_integrated_stations
from data_pipeline.generate_parcels import calculate_drone_feasible, generate_parcels
from data_pipeline.parse_bus_route import load_or_generate_bus_route
from data_pipeline.synthesize_timetable import synthesize_timetable
from experiments.smoke_test_data_pipeline import run_smoke_test
from utils.config import load_config

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "shanghai_small.yaml"


@pytest.fixture
def config() -> dict:
    return load_config(CONFIG_PATH)


def test_fallback_graph_creation_is_deterministic(config: dict) -> None:
    first = create_fallback_graph(config)
    second = create_fallback_graph(config)
    assert first == second
    assert len(first.nodes) == 25
    assert len(first.edges) == 80
    assert all({"node_id", "x", "y"} <= set(node) for node in first.nodes.values())
    assert all({"from_node", "to_node", "length_m", "speed_kph", "travel_time_min"} <= set(edge) for edge in first.edges)


def test_timetable_is_monotonic(config: dict, tmp_path: Path) -> None:
    graph = create_fallback_graph(config)
    stops = load_or_generate_bus_route(config, graph, tmp_path)
    trips, stop_times, _ = synthesize_timetable(stops, config, tmp_path)
    assert 15 <= len(stops) <= 30
    for trip in trips:
        times = [row for row in stop_times if row["trip_id"] == trip["trip_id"]]
        arrivals = [row["arrival_time"] for row in times]
        assert arrivals == sorted(arrivals)
        assert all(row["arrival_time"] <= row["departure_time"] for row in times)


def test_parcel_deadlines_and_required_mappings(config: dict, tmp_path: Path) -> None:
    graph = create_fallback_graph(config)
    stops = load_or_generate_bus_route(config, graph, tmp_path)
    stations = select_integrated_stations(stops, config)
    parcels = generate_parcels(config, graph, stations, tmp_path)
    assert all(parcel["release_time"] < parcel["deadline"] for parcel in parcels)
    assert all(parcel["nearest_station_id"] and parcel["nearest_road_node"] for parcel in parcels)
    assert all(isinstance(parcel["drone_feasible"], bool) for parcel in parcels)
    ranges = {"tight": (20, 40), "moderate": (40, 80), "loose": (80, 140)}
    for parcel in parcels:
        low, high = ranges[parcel["deadline_type"]]
        assert low <= parcel["deadline"] - parcel["release_time"] <= high


def test_drone_feasibility_uses_all_three_constraints(config: dict) -> None:
    assert calculate_drone_feasible(1.0, 31.30, 121.50, 31.301, 121.501, config)
    assert not calculate_drone_feasible(config["network"]["drone_payload_kg"] + 0.1, 31.30, 121.50, 31.301, 121.501, config)
    assert not calculate_drone_feasible(1.0, 31.30, 121.50, 31.40, 121.60, config)
    time_limited = json.loads(json.dumps(config))
    time_limited["network"]["max_drone_round_trip_min"] = 0.01
    assert not calculate_drone_feasible(1.0, 31.30, 121.50, 31.301, 121.501, time_limited)


def test_integrated_station_selection_includes_endpoints(config: dict, tmp_path: Path) -> None:
    graph = create_fallback_graph(config)
    stops = load_or_generate_bus_route(config, graph, tmp_path)
    stations = select_integrated_stations(stops, config)
    assert len(stations) == config["network"]["num_integrated_stations"]
    assert stations[0]["stop_id"] == stops[0]["stop_id"]
    assert stations[-1]["stop_id"] == stops[-1]["stop_id"]
    assert len({station["stop_id"] for station in stations}) == len(stations)


def test_instance_files_load_and_smoke_test_is_offline(config: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("Fallback Stage 2 pipeline attempted network access")

    monkeypatch.setattr(socket, "socket", reject_network)
    result = run_smoke_test(CONFIG_PATH, fallback=True, output_root=tmp_path)
    output_dir = Path(result["output_directory"])
    assert all((output_dir / filename).is_file() for filename in REQUIRED_FILENAMES)
    assert json.loads((output_dir / "instance.json").read_text())["mode"] == "fallback"
    assert load_config(output_dir / "instance.yaml")["stage"] == 2


def test_build_instance_is_reproducible(config: dict, tmp_path: Path) -> None:
    first = build_instance(CONFIG_PATH, fallback=True, output_root=tmp_path / "one")
    second = build_instance(CONFIG_PATH, fallback=True, output_root=tmp_path / "two")
    assert first["counts"] == second["counts"]
    one = Path(first["output_directory"])
    two = Path(second["output_directory"])
    for filename in REQUIRED_FILENAMES:
        if filename not in {"instance.json", "instance.yaml"}:
            assert (one / filename).read_bytes() == (two / filename).read_bytes()
