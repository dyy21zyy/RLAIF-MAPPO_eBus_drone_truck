"""Original-scale real-transit data-setting tests.

These tests specify the public behavior for the research data setting:
real stop/stop-time inputs where available, original eBus-Drone scale and
system defaults where inherited, explicit truck-extension assumptions, and
source-aware outputs. They intentionally do not require external web access.
"""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from data_pipeline.build_instance import build_instance
from data_pipeline.common import read_csv
from data_pipeline.original_ebus_drone import load_original_ebus_drone_defaults
from data_pipeline.real_transit import load_simplified_transit_csv
from envs import DynamicDeliveryEnv
from rlaif.collect_assignment_states import build_assignment_state
from rlaif.prompt_builder import build_prompt_records, select_action_pairs
from utils.config import load_config

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "original_scale_real_transit.yaml"
DEFAULTS_PATH = ROOT / "configs" / "original_ebus_drone_defaults.yaml"
TRANSIT_FIXTURES = ROOT / "tests" / "fixtures" / "transit"


def _fixture_config(tmp_path: Path, *, allow_synthetic_stop_times: bool = False) -> Path:
    config = deepcopy(load_config(CONFIG_PATH))
    config["reference"]["defaults_config"] = str(DEFAULTS_PATH)
    config["project"]["output_dir"] = str(tmp_path / "outputs")
    config["project"]["log_dir"] = str(tmp_path / "logs")
    config["project"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
    config["city"]["name"] = "fixture_original_scale_real_transit"
    config["transit"]["stops_csv"] = str(TRANSIT_FIXTURES / "real_bus_stops.csv")
    config["transit"]["trips_csv"] = str(TRANSIT_FIXTURES / "real_bus_trips.csv")
    config["transit"]["stop_times_csv"] = str(TRANSIT_FIXTURES / "real_bus_stop_times.csv")
    config["transit"]["allow_synthetic_timetable_if_missing"] = allow_synthetic_stop_times
    path = tmp_path / "fixture_original_scale_real_transit.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def test_original_ebus_drone_defaults_config_contains_required_scale_and_sources() -> None:
    defaults = load_original_ebus_drone_defaults(DEFAULTS_PATH)

    assert defaults["scale"]["num_stops"] == 30
    assert defaults["scale"]["num_integrated_stations"] == 8
    assert defaults["scale"]["num_parcels"] == 60
    assert defaults["scale"]["service_horizon_min"] == 480
    assert defaults["bus"]["planned_headway_min"] == 10
    assert defaults["drone"]["speed_kmph"] == 40.0
    assert defaults["parcel"]["locker_capacity_kg"] == 30.0
    assert defaults["sources"]["scale.num_stops"]["source_type"] == "original_ebus_drone"


def test_original_scale_real_transit_config_loads_with_required_policy_sections() -> None:
    config = load_config(CONFIG_PATH)

    assert config["data_mode"] == "original_scale_real_transit"
    assert config["scale_policy"]["match_original_ebus_drone"] is True
    assert config["transit"]["allow_synthetic_timetable_if_missing"] is False
    assert config["truck_extension"]["source"] == "explicit_new_truck_extension"


def test_simplified_real_transit_csv_loader_preserves_stop_sequence_and_stop_times() -> None:
    transit = load_simplified_transit_csv(
        TRANSIT_FIXTURES / "real_bus_stops.csv",
        TRANSIT_FIXTURES / "real_bus_trips.csv",
        TRANSIT_FIXTURES / "real_bus_stop_times.csv",
    )

    assert [row["stop_sequence"] for row in transit["stops"]] == list(range(1, 25))
    assert transit["trips"][0]["trip_id"] == "real_trip_001"
    assert transit["stop_times"][0]["source_type"] == "real_transit_data"
    arrivals = [
        row["arrival_time"]
        for row in transit["stop_times"]
        if row["trip_id"] == "real_trip_001"
    ]
    assert arrivals == sorted(arrivals)


def test_original_scale_real_transit_build_writes_provenance_and_scale_report(tmp_path: Path) -> None:
    config_path = _fixture_config(tmp_path)
    instance = build_instance(config_path, fallback=False, output_root=tmp_path / "processed")
    output_dir = Path(instance["output_directory"])

    provenance = json.loads((output_dir / "data_provenance.json").read_text(encoding="utf-8"))
    scale = json.loads((output_dir / "scale_match_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "instance.json").read_text(encoding="utf-8"))

    assert manifest["mode"] == "original_scale_real_transit"
    assert manifest["counts"]["bus_stops"] == 24
    assert manifest["counts"]["integrated_stations"] == 8
    assert manifest["counts"]["parcels"] == 60
    assert scale["original_num_stops"] == 30
    assert scale["new_num_stops"] == 24
    assert scale["original_num_integrated_stations"] == 8
    assert scale["new_num_integrated_stations"] == 8
    assert scale["original_num_parcels"] == scale["new_num_parcels"] == 60
    assert scale["scale_match_pass"] is True
    assert {entry["field"]: entry["source_type"] for entry in provenance["entries"]}["bus stop_times"] == "real_transit_data"


def test_missing_stop_times_requires_explicit_synthesis_and_marks_original_source(tmp_path: Path) -> None:
    config_path = _fixture_config(tmp_path, allow_synthetic_stop_times=False)
    config = load_config(config_path)
    missing = tmp_path / "missing_stop_times.csv"
    config["transit"]["stop_times_csv"] = str(missing)
    blocked = tmp_path / "blocked.json"
    blocked.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="stop_times"):
        build_instance(blocked, fallback=False, output_root=tmp_path / "blocked")

    config["transit"]["allow_synthetic_timetable_if_missing"] = True
    allowed = tmp_path / "allowed.json"
    allowed.write_text(json.dumps(config), encoding="utf-8")
    instance = build_instance(allowed, fallback=False, output_root=tmp_path / "allowed")
    provenance = json.loads((Path(instance["output_directory"]) / "data_provenance.json").read_text(encoding="utf-8"))
    sources = {entry["field"]: entry for entry in provenance["entries"]}

    assert sources["bus stop_times"]["source_type"] == "original_ebus_drone"
    assert "real stop_times unavailable" in sources["bus stop_times"]["notes"]


def test_integrated_stations_parcels_and_bus_events_use_real_transit_inputs(tmp_path: Path) -> None:
    instance = build_instance(_fixture_config(tmp_path), fallback=False, output_root=tmp_path / "processed")
    output_dir = Path(instance["output_directory"])

    stops = {row["stop_id"] for row in read_csv(output_dir / "bus_stops.csv")}
    stations = read_csv(output_dir / "integrated_stations.csv")
    parcels = read_csv(output_dir / "parcels.csv")
    assert {row["stop_id"] for row in stations} <= stops
    assert len(stations) == 8
    assert len(parcels) == 60

    env = DynamicDeliveryEnv(output_dir / "instance.json")
    observation, _info = env.reset(seed=123)
    bus_event_times = sorted(
        event.time_min
        for event in env.events
        if event.kind == "bus_arrival"
    )
    fixture_first_station_stop = stations[0]["stop_id"]
    expected_first_arrivals = sorted(
        float(row["arrival_time"])
        for row in read_csv(output_dir / "bus_stop_times.csv")
        if row["stop_id"] == fixture_first_station_stop
    )
    observed_times = []
    if observation["agent"] == "bus":
        observed_times.append(float(observation["time_min"]))
    observed_times.extend(bus_event_times[: len(expected_first_arrivals)])
    assert sorted(observed_times)[: len(expected_first_arrivals)] == expected_first_arrivals
    assert observation["agent"] in {"assignment", "bus", "terminal"}


def test_tbd_feasibility_and_rlaif_prompts_are_source_aware(tmp_path: Path) -> None:
    instance = build_instance(_fixture_config(tmp_path), fallback=False, output_root=tmp_path / "processed")
    env = DynamicDeliveryEnv(Path(instance["output_directory"]) / "instance.json")
    observation, _info = env.reset(seed=321)

    while observation["agent"] != "terminal" and observation["agent"] != "assignment":
        observation, *_ = env.step(0)

    assert observation["agent"] == "assignment"
    state = build_assignment_state(env, episode_id=0)
    assert state["data_sources"]["bus_stop_times"]["source_type"] == "real_transit_data"
    tbd_actions = [
        action
        for action in state["candidate_actions"]
        if action["action_name"].startswith("TBD_")
    ]
    assert tbd_actions
    assert any(action["feasible"] for action in tbd_actions)

    paired = state if select_action_pairs(state) else None
    assert paired is not None
    prompt = build_prompt_records([paired])[0]
    assert "real_transit_data" in prompt["prompt_text"]
    assert "Do not treat inherited or fallback fields as real-world observations" in prompt["prompt_text"]
    assert "chosen" not in prompt
    assert "rejected" not in prompt


def test_generated_data_artifact_patterns_are_ignored() -> None:
    probes = [
        "data/processed/example/instance.json",
        "data/raw/transit/real_bus_stops.csv",
        "data/preference/example.jsonl",
        "results/example.json",
        "runs/example.log",
        "logs/loop_engineering/hardening_run.log",
        "checkpoints/reward_model.pt",
        "model.ckpt",
    ]
    for probe in probes:
        completed = subprocess.run(
            ["git", "check-ignore", "-q", probe],
            text=True,
            cwd=ROOT,
            capture_output=True,
        )
        assert completed.returncode == 0, probe
