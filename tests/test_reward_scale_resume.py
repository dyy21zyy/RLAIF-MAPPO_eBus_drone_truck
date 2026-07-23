import json
import time
from pathlib import Path

import pytest
import yaml

from envs.reward_components import REWARD_COMPONENTS
from envs.reward_scales import canonical_payload_hash
from evaluation.scenario_bank import ScenarioBank, FrozenScenario, sha256_json
from evaluation.reward_scale_reference_policies import ReferencePolicy
from experiments import estimate_reward_reference_scales as ers


def _scenario(tmp_path, sid):
    inst = tmp_path / f"{sid}.json"; inst.write_text("{}\n")
    return FrozenScenario(sid, "train", str(inst), str(inst), {}, {}, "cfg", f"content-{sid}")


def _config(tmp_path, bank_hash="bank-a"):
    cfg = {
        "scenario_bank": {"expected_bank_hash": bank_hash},
        "reference_policies": ["p1"],
        "minimum_valid_episodes": 1,
        "maximum_failure_fraction": 0.0,
        "estimator": {"method": "percentile", "percentile": 95},
    }
    p = tmp_path / "cfg.yaml"; p.write_text(yaml.safe_dump(cfg))
    return p, cfg


def _patch_bank(monkeypatch, tmp_path, bank_hash="bank-a", count=2):
    scenarios = tuple(_scenario(tmp_path, f"s{i}") for i in range(count))
    manifest = {"split": "train", "bank_hash": bank_hash, "scenario_count": count}
    bank = ScenarioBank("b", "train", scenarios, bank_hash)
    monkeypatch.setattr(ers, "validate_training_bank", lambda *a, **k: (manifest, bank))
    monkeypatch.setattr(ers, "get_reference_policies", lambda names=None: [ReferencePolicy("p1", 1)])
    return manifest, bank


def _success_row(s, cfg, bank_hash="bank-a"):
    resolved = json.loads(json.dumps(cfg, sort_keys=True, default=str))
    resolved["run_classification"] = "formal"
    resolved["scenario_bank"] = {**resolved.get("scenario_bank", {}), "manifest": "bank", "bank_hash": bank_hash}
    row = {"scenario_id": s.scenario_id, "scenario_content_hash": s.scenario_content_hash, "instance_hash": s.instance_hash, "scenario_bank_hash": bank_hash, "reference_policy": "p1", "reference_policy_version": 1, "estimation_seed": 0, "resolved_config_hash": sha256_json(resolved), "run_classification": "formal", "episode_status": "success", "transition_count": 1, "runtime": 0.0, "released_parcels": 1, "delivered_parcels": 1, "failure_reason": "", "exception_type": ""}
    for c in REWARD_COMPONENTS: row[f"raw_{c}"] = 1.0
    return row


def _artifact(path, bank_hash="bank-a", status="passed", omit_component=None, scale=1.0, classification="formal"):
    comps = {c: {"scale": scale, "status": "observed_positive", "positive_count": 1, "minimum_override": None} for c in REWARD_COMPONENTS if c != omit_component}
    scales = {c: scale for c in REWARD_COMPONENTS if c != omit_component}
    payload = {"artifact_type": "reward_reference_scales", "artifact_version": 1, "run_classification": classification, "validation_status": status, "component_order": list(REWARD_COMPONENTS), "training_scenario_bank_hash": bank_hash, "components": comps, "scales": scales, "estimator": {"method": "percentile", "percentile": 95}}
    payload["artifact_hash"] = canonical_payload_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, sort_keys=True))
    return payload


def test_resume_reuses_valid_completed_artifact(tmp_path, monkeypatch):
    _patch_bank(monkeypatch, tmp_path)
    cfg_path, _ = _config(tmp_path)
    out = tmp_path / "out" / "scale.json"; _artifact(out)
    before = out.stat().st_mtime_ns
    monkeypatch.setattr(ers, "DynamicDeliveryEnv", lambda *a, **k: pytest.fail("rollout executed"))
    assert ers.run_estimation("bank", cfg_path, out, run_classification="formal", resume=True)["training_scenario_bank_hash"] == "bank-a"
    assert out.stat().st_mtime_ns == before


def test_resume_partial_progress_runs_only_unfinished_without_duplicates(tmp_path, monkeypatch):
    _, bank = _patch_bank(monkeypatch, tmp_path, count=2)
    cfg_path, cfg = _config(tmp_path)
    out = tmp_path / "out" / "scale.json"; out.parent.mkdir()
    (out.parent / "reward_scale_episode_components.progress.jsonl").write_text(json.dumps(_success_row(bank.scenarios[0], cfg)) + "\n")
    class Env:
        def __init__(self, path): self.config={}; self.raw_cost_components={c: 2.0 for c in REWARD_COMPONENTS}; self.parcels={1:1}; self.done=False
        def reset(self, seed=0): return {"agent_id":"a", "action_mask":[1]}, {}
        def step(self, action):
            self.done=True; return {"agent_id":"terminal"}, 0, True, False, {"delivered_parcels": 1, "raw_cost_components": self.raw_cost_components}
    monkeypatch.setattr(ers, "DynamicDeliveryEnv", Env)
    artifact = ers.run_estimation("bank", cfg_path, out, run_classification="formal", resume=True)
    rows = [json.loads(l) for l in (out.parent / "reward_scale_episode_components.jsonl").read_text().splitlines()]
    assert artifact["validation_status"] == "passed"
    assert [(r["reference_policy"], r["scenario_id"]) for r in rows] == [("p1", "s0"), ("p1", "s1")]


def test_resume_rejects_wrong_bank_progress(tmp_path, monkeypatch):
    _, bank = _patch_bank(monkeypatch, tmp_path)
    cfg_path, cfg = _config(tmp_path)
    out = tmp_path / "out" / "scale.json"; out.parent.mkdir()
    bad = _success_row(bank.scenarios[0], cfg); bad["scenario_bank_hash"] = "other"
    (out.parent / "reward_scale_episode_components.progress.jsonl").write_text(json.dumps(bad) + "\n")
    with pytest.raises(ers.RewardScaleEstimationError, match="scenario_bank_hash mismatch"):
        ers.run_estimation("bank", cfg_path, out, run_classification="formal", resume=True)


@pytest.mark.parametrize("kwargs", [
    {"status": "blocked"},
    {"omit_component": REWARD_COMPONENTS[0]},
    {"scale": 0.0},
    {"classification": "diagnostic"},
])
def test_resume_rejects_invalid_completed_artifact(tmp_path, monkeypatch, kwargs):
    _patch_bank(monkeypatch, tmp_path)
    cfg_path, _ = _config(tmp_path)
    out = tmp_path / "out" / "scale.json"; _artifact(out, **kwargs)
    with pytest.raises(ValueError):
        ers.run_estimation("bank", cfg_path, out, run_classification="formal", resume=True)


def test_existing_output_refuses_without_resume_or_force(tmp_path, monkeypatch):
    _patch_bank(monkeypatch, tmp_path); cfg_path, _ = _config(tmp_path)
    out = tmp_path / "out" / "scale.json"; _artifact(out)
    with pytest.raises(FileExistsError): ers.run_estimation("bank", cfg_path, out, run_classification="formal")


def test_force_rebuilds_existing_output(tmp_path, monkeypatch):
    _patch_bank(monkeypatch, tmp_path, count=1); cfg_path, _ = _config(tmp_path)
    out = tmp_path / "out" / "scale.json"; _artifact(out, scale=3.0); time.sleep(0.001)
    class Env:
        def __init__(self, path): self.config={}; self.raw_cost_components={c: 2.0 for c in REWARD_COMPONENTS}; self.parcels={1:1}
        def reset(self, seed=0): return {"agent_id":"a", "action_mask":[1]}, {}
        def step(self, action): return {"agent_id":"terminal"}, 0, True, False, {"delivered_parcels": 1, "raw_cost_components": self.raw_cost_components}
    monkeypatch.setattr(ers, "DynamicDeliveryEnv", Env)
    assert all(v == 2.0 for v in ers.run_estimation("bank", cfg_path, out, run_classification="formal", force=True)["scales"].values())


def test_resume_force_invalid_combination(tmp_path, monkeypatch):
    _patch_bank(monkeypatch, tmp_path); cfg_path, _ = _config(tmp_path)
    with pytest.raises(ValueError): ers.run_estimation("bank", cfg_path, tmp_path/"x.json", resume=True, force=True)
