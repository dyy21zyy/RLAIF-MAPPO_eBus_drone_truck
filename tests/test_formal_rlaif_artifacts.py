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
