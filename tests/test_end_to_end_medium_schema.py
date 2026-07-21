from pathlib import Path
from utils.config import load_config


def test_medium_schema_contains_formal_components():
    cfg = load_config(Path(__file__).parents[1] / "configs/paper/base_medium.yaml")
    assert cfg["scenario"]["size"] == "medium"
    assert cfg["truck"]["weight_capacity_kg"] > 0
    assert cfg["truck"]["volume_capacity_m3"] > 0
    assert cfg["bus"]["passenger_capacity"] > 0
    assert cfg["drone_battery"]["charging_start_is_learned"] is True
