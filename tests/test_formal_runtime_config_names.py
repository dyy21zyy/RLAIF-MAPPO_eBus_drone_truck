from __future__ import annotations

import json
from pathlib import Path

import yaml

from envs.reward_components import REWARD_COMPONENTS
from experiments import prepare_formal_experiment_inputs as prep
from evaluation.formal_launch_plan import generate_formal_launch_plan


def _install_fast_formal_generators(monkeypatch):
    def fake_build_bank(base, split, count, seed_start, output, **kwargs):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "scenario_bank_manifest.json").write_text(
            json.dumps(
                {
                    "split": split,
                    "scenario_count": count,
                    "bank_hash": f"{split}-canonical-hash",
                    "scenarios": [],
                }
            )
            + "\n"
        )

    def fake_run_estimation(bank_path, config_path, output_path, **kwargs):
        artifact = {
            "artifact_hash": "reward-scale-canonical-hash",
            "training_scenario_bank_hash": "train-canonical-hash",
            "estimator": "test-fast-estimator",
            "components": {
                component: {"scale": 1.0, "status": "observed_positive"}
                for component in REWARD_COMPONENTS
            },
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact) + "\n")
        return artifact

    monkeypatch.setattr(prep, "build_bank", fake_build_bank)
    monkeypatch.setattr(prep, "run_estimation", fake_run_estimation)


def test_prepare_writes_canonical_baseline_runtime_config_names(tmp_path, monkeypatch):
    _install_fast_formal_generators(monkeypatch)

    manifest = prep.prepare(
        tmp_path,
        force=True,
        counts={"train": 1, "validation": 1, "test": 1},
        scale_scenario_limit=1,
    )

    configs = tmp_path / "configs"
    assert (configs / "assignment_ppo.yaml").is_file()
    assert (configs / "mappo_env.yaml").is_file()
    assert not (configs / "train_assignment_ppo.resolved.yaml").exists()
    assert not (configs / "train_mappo_env.resolved.yaml").exists()
    assert (configs / "mappo_rlaif_assignment.yaml").is_file()
    assert (configs / "mappo_rlaif_all.yaml").is_file()
    assert manifest["resolved_configs"]["train_assignment_ppo"] == str(configs / "assignment_ppo.yaml")
    assert manifest["resolved_configs"]["train_mappo_env"] == str(configs / "mappo_env.yaml")

    for filename in ("assignment_ppo.yaml", "mappo_env.yaml"):
        text = (configs / filename).read_text()
        cfg = yaml.safe_load(text)
        assert not prep.PLACEHOLDER_RE.search(text)
        assert cfg["env"]["expected_train_bank_hash"] == "train-canonical-hash"
        assert cfg["env"]["expected_validation_bank_hash"] == "validation-canonical-hash"
        assert cfg["env"]["expected_test_bank_hash"] == "test-canonical-hash"
        assert cfg["reward"]["scale_artifact_hash"] == "reward-scale-canonical-hash"
        assert cfg["reward"]["expected_training_scenario_bank_hash"] == "train-canonical-hash"
        assert cfg["run_classification"] == "formal"
        assert cfg["env"]["fallback"] is False


def test_resume_and_force_preserve_canonical_runtime_config_names(tmp_path, monkeypatch):
    _install_fast_formal_generators(monkeypatch)
    counts = {"train": 1, "validation": 1, "test": 1}

    prep.prepare(tmp_path, force=True, counts=counts, scale_scenario_limit=1)
    prep.prepare(tmp_path, resume=True, counts=counts, scale_scenario_limit=1)
    prep.prepare(tmp_path, force=True, counts=counts, scale_scenario_limit=1)

    manifest = json.loads((tmp_path / "formal_input_manifest.json").read_text())
    assert manifest["resolved_configs"]["train_assignment_ppo"].endswith("/configs/assignment_ppo.yaml")
    assert manifest["resolved_configs"]["train_mappo_env"].endswith("/configs/mappo_env.yaml")
    assert (tmp_path / "configs" / "assignment_ppo.yaml").is_file()
    assert (tmp_path / "configs" / "mappo_env.yaml").is_file()
    assert not (tmp_path / "configs" / "train_assignment_ppo.resolved.yaml").exists()
    assert not (tmp_path / "configs" / "train_mappo_env.resolved.yaml").exists()


def test_baseline_runtime_content_matches_legacy_resolution_except_filename(tmp_path):
    manifests = {s: {"bank_hash": f"{s}hash", "scenario_count": 1} for s in ("train", "validation", "test")}
    scale = {"artifact_hash": "scalehash", "training_scenario_bank_hash": "trainhash"}

    for template, canonical in (
        ("train_assignment_ppo", "assignment_ppo.yaml"),
        ("train_mappo_env", "mappo_env.yaml"),
    ):
        src = Path("configs/paper") / f"{template}.yaml"
        old = prep._resolve_config(src, tmp_path / "old" / f"{template}.resolved.yaml", manifests, scale, tmp_path, {})
        new = prep._resolve_config(src, tmp_path / "new" / canonical, manifests, scale, tmp_path, {})
        assert yaml.safe_load(new.read_text()) == yaml.safe_load(old.read_text())


def test_formal_launch_plan_and_runbook_use_canonical_baseline_paths(tmp_path):
    plan = generate_formal_launch_plan(
        {"training_seeds": [1], "input_hashes": {"scenario_bank_hash": "readyhash"}},
        tmp_path / "launch_plan.json",
    )
    commands = "\n".join(stage.get("command") or "" for stage in plan["stages"])
    runbook = Path("docs/FORMAL_RLAIF_RUNBOOK.md").read_text()
    combined = commands + "\n" + runbook

    assert "results/formal/configs/assignment_ppo.yaml" in combined
    assert "results/formal/configs/mappo_env.yaml" in combined
    assert "train_assignment_ppo.resolved.yaml" not in combined
    assert "train_mappo_env.resolved.yaml" not in combined
