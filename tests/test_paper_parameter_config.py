from pathlib import Path
import yaml

from envs.config_schema import validate_paper_config, validate_mappo_config

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_formal_paper_configs_validate():
    for name in ["base_small.yaml", "base_medium.yaml", "base_large.yaml"]:
        validate_paper_config(load(f"configs/paper/{name}"))


def test_small_medium_large_counts_are_consistent():
    expected = {
        "base_small.yaml": (30, 6, 24, 8, 15),
        "base_medium.yaml": (60, 8, 36, 12, 10),
        "base_large.yaml": (90, 8, 45, 16, 8),
    }
    for name, values in expected.items():
        cfg = load(f"configs/paper/{name}")
        assert (cfg["scenario"]["num_parcels"], cfg["network"]["num_integrated_stations"], cfg["bus_schedule"]["scheduled_trip_count"], cfg["bus_schedule"]["freight_trip_count"], cfg["bus_schedule"]["planned_headway_min"]) == values
        assert len(cfg["network"]["integrated_station_stop_indices"]) == cfg["network"]["num_integrated_stations"]


def test_formal_mappo_config_is_not_smoke_config():
    for name in ["train_mappo_env.yaml", "train_mappo_rlaif.yaml"]:
        cfg = load(f"configs/paper/{name}")
        validate_mappo_config(cfg)
        assert cfg["training"]["total_episodes"] == 3000
        forbidden = {"epsilon", "epsilon_greedy", "replay_buffer", "target_q_network", "polyak", "fixed_nine_action_output"}
        assert not (forbidden & set(cfg["training"]))
