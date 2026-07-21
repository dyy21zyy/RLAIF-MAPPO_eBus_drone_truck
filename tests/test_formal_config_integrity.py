from pathlib import Path
from utils.config import load_config

ROOT = Path(__file__).parents[1]


def test_formal_rlaif_fail_closed_and_four_agents():
    cfg = load_config(ROOT / "configs/paper/train_mappo_rlaif.yaml")
    r = cfg["rlaif"]
    assert r["fallback_to_env_reward"] is False
    assert r["fail_on_invalid_reward_model"] is True
    assert set(r["agents"]) == {"assignment", "truck", "bus", "station"}


def test_formal_configs_have_no_assignment_only_reward_path_or_smoke_checkpoints():
    for rel in ["benchmark.yaml", "ablation.yaml", "sensitivity.yaml", "train_mappo_env.yaml", "train_mappo_rlaif.yaml"]:
        text = (ROOT / "configs/paper" / rel).read_text(encoding="utf-8").lower()
        assert "assignment-only" not in text
        assert "smoke checkpoint" not in text
        assert "smoke_checkpoint" not in text


def test_scenario_banks_disjoint():
    cfg = load_config(ROOT / "configs/paper/scenario_banks.yaml")
    offsets = [bank["seed_offset"] for bank in cfg["banks"].values()]
    assert len(offsets) == len(set(offsets))
