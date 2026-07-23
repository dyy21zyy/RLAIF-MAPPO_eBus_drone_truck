import json
import os
import pytest

from experiments.prepare_formal_rlaif_artifacts import cache_identity, validate_structured_response, verify_evaluator_config
from rlaif.grouped_split import grouped_split


def test_malformed_evaluator_output_rejected():
    with pytest.raises(ValueError, match="non-JSON"):
        validate_structured_response("not json")
    with pytest.raises(ValueError, match="unknown preferred"):
        validate_structured_response(json.dumps({"preferred": "C", "confidence": 0.8, "criteria": {}, "reason": "ok"}))
    with pytest.raises(ValueError, match="hidden method"):
        validate_structured_response(json.dumps({"preferred": "A", "confidence": 0.8, "criteria": {"timeliness": "A"}, "reason": "MAPPO policy wins"}))


def test_evaluator_cache_identity_includes_model_and_prompt():
    base = dict(agent_type="bus", scenario_hash="s", transition_pair_hash="p", prompt_version="v1", schema_version="schema1", evaluator_model="m1", evaluator_parameters={"temperature": 0.0})
    assert cache_identity(**base) != cache_identity(**{**base, "evaluator_model": "m2"})
    assert cache_identity(**base) != cache_identity(**{**base, "prompt_version": "v2"})


def test_missing_api_credential_failure(monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        verify_evaluator_config({"evaluator": {"api_key_env": "OPENAI_API_KEY", "base_url_env": "OPENAI_BASE_URL", "model_env": "OPENAI_MODEL"}})


def test_formal_preference_split_isolation_by_scenario():
    records = [{"scenario_id": f"s{i}", "episode_id": "e", "state_id": "st"} for i in range(12)]
    split = grouped_split(records, .5, .25, .25, seed=1, group_by="scenario")
    train = {tuple(x) for x in split["splits"]["train"]}
    val = {tuple(x) for x in split["splits"]["validation"]}
    test = {tuple(x) for x in split["splits"]["test"]}
    assert not (train & val or train & test or val & test)


def test_prepare_uses_canonical_train_bank_hash_without_undefined_fallbacks(tmp_path, monkeypatch):
    import yaml
    from pathlib import Path
    from experiments import prepare_formal_rlaif_artifacts as prep

    bank_dir = tmp_path / "bank"; bank_dir.mkdir()
    train_manifest = bank_dir / "scenario_bank_manifest.json"
    train_manifest.write_text(json.dumps({"split": "train", "scenario_count": 1, "bank_hash": "canonical-bank", "scenarios": []}) + "\n")
    output_root = tmp_path / "rlaif"
    agents = {}
    for agent in prep.AGENT_TYPES:
        pref = tmp_path / f"{agent}.jsonl"; pref.write_text("{}\n")
        ckpt = tmp_path / f"{agent}.pt"
        agents[agent] = {"output_preferences": str(pref), "target_valid_pair_count": 1, "reward_checkpoint": str(ckpt), "supported_event_types": ["event"]}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "run_classification": "formal",
        "scenario_bank": {"final_train_manifest": str(train_manifest)},
        "evaluator": {"api_key_env": "OPENAI_API_KEY", "base_url_env": "OPENAI_BASE_URL", "model_env": "OPENAI_MODEL", "prompt_version": "v1"},
        "agents": agents,
        "reward_model": {"config_template": {}},
        "outputs": {"manifest": str(tmp_path / "artifact_manifest.json")},
    }))
    monkeypatch.setenv("OPENAI_API_KEY", "k"); monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid"); monkeypatch.setenv("OPENAI_MODEL", "m")
    monkeypatch.setattr(prep, "generate_preferences", lambda *a, **k: None)
    monkeypatch.setattr(prep, "_validate_preference_file", lambda agent, path, target: {"rows": 1, "usable_binary": 1, "counts_by_event": {}})
    def fake_train(argv):
        Path(argv[argv.index("--output") + 1]).write_text("ckpt")
        return 0
    monkeypatch.setattr(prep, "train_reward_main", fake_train)
    monkeypatch.setattr(prep, "_validate_checkpoint", lambda *a, **k: None)
    def fake_inject(template_path, output_path, manifest, scope_agents, cfg):
        local_manifest = prep.load_bank_manifest(Path(cfg["scenario_bank"]["final_train_manifest"]))
        manifest_file_hash = prep.sha256_file(Path(cfg["scenario_bank"]["final_train_manifest"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml.safe_dump({"env": {"expected_bank_hash": local_manifest["bank_hash"], "scenario_bank_manifest_file_hash": manifest_file_hash}, "reward": {"expected_training_scenario_bank_hash": local_manifest["bank_hash"]}}))
    monkeypatch.setattr(prep, "_inject_artifacts", fake_inject)

    manifest = prep.prepare(cfg_path, output_root, resume=True)
    runtime = yaml.safe_load((tmp_path / "configs" / "mappo_rlaif_all.yaml").read_text())
    assert manifest["scenario_bank"]["bank_hash"] == "canonical-bank"
    assert manifest["scenario_bank"]["manifest_file_hash"] != "canonical-bank"
    assert runtime["env"]["expected_bank_hash"] == "canonical-bank"
