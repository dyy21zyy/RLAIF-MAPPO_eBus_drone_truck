from pathlib import Path
import yaml

from envs.config_schema import flatten_parameter_keys, validate_parameter_provenance

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_all_medium_parameters_have_provenance():
    params = flatten_parameter_keys(load("configs/paper/base_medium.yaml"))
    provenance = load("configs/paper/parameter_provenance.yaml")
    validate_parameter_provenance(params, provenance)


def test_required_project_extensions_are_marked():
    provenance = load("configs/paper/parameter_provenance.yaml")
    required = [
        "truck.num_trucks", "truck.weight_capacity_kg", "truck.volume_capacity_m3",
        "parcel.volume_min_m3", "parcel.volume_max_m3", "bus_schedule.minimum_layover_min",
        "bus_schedule.relocation_time_min", "truck.loading_time_min", "truck.unloading_time_min",
        "truck.cost_per_km",
    ]
    for key in required:
        assert provenance[key]["category"] == "project_extension"
