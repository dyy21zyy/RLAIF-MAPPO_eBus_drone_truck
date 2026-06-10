"""Stage 4 RLAIF data workflow regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlaif.ai_evaluator import parse_ai_response, run_api, run_offline, run_replay
from rlaif.collect_assignment_states import collect_assignment_states
from rlaif.preference_dataset import (CANDIDATE_FEATURE_KEYS, read_jsonl, validate_assignment_state,
                                      validate_prompt, write_jsonl)
from rlaif.prompt_builder import build_prompt_records, select_action_pairs

CONFIG = Path(__file__).parents[1] / "configs/shanghai_small.yaml"


def paired_state(states: list[dict[str, object]]) -> dict[str, object]:
    return next(state for state in states if select_action_pairs(state))


@pytest.fixture(scope="module")
def states(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, object]]:
    output = tmp_path_factory.mktemp("rlaif") / "states.jsonl"
    return collect_assignment_states(CONFIG, 1, output, fallback=True)


def test_assignment_state_schema_validation(states: list[dict[str, object]]) -> None:
    assert len(states) == 60
    validate_assignment_state(states[0])
    assert len(states[0]["candidate_actions"]) == 13


def test_candidate_action_feature_validation(states: list[dict[str, object]]) -> None:
    for features in states[0]["candidate_action_features"].values():
        assert CANDIDATE_FEATURE_KEYS <= features.keys()
        assert isinstance(features["infeasibility_reasons"], list)
        assert "preference_score" not in features


def test_prompt_json_requirement_check(states: list[dict[str, object]]) -> None:
    prompt = build_prompt_records([paired_state(states)])[0]
    validate_prompt(prompt)
    assert "Return only valid JSON" in prompt["prompt_text"]
    assert '"chosen"' in prompt["prompt_text"]


def test_pair_selection_logic(states: list[dict[str, object]]) -> None:
    pairs = select_action_pairs(paired_state(states))
    assert 1 <= len(pairs) <= 3
    assert any("TD" in pair for pair in pairs)
    assert len(pairs) == len(set(pairs))


def test_invalid_ai_response_handling(states: list[dict[str, object]]) -> None:
    prompt = build_prompt_records([paired_state(states)])[0]
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_ai_response(prompt, "not json", "test-model", 0.0)
    invalid = json.dumps({"chosen": "UNKNOWN", "rejected": prompt["action_a"], "confidence": 0.9, "reason": "x"})
    with pytest.raises(ValueError, match="chosen"):
        parse_ai_response(prompt, invalid, "test-model", 0.0)


def test_low_confidence_label_marked_unusable(states: list[dict[str, object]]) -> None:
    prompt = build_prompt_records([paired_state(states)])[0]
    raw = json.dumps({"chosen": prompt["action_a"], "rejected": prompt["action_b"], "confidence": 0.59, "reason": "uncertain"})
    record = parse_ai_response(prompt, raw, "test-model", 0.0)
    assert record["validation_status"] == "valid"
    assert record["usable_for_training"] is False


def test_replay_mode_validation(states: list[dict[str, object]], tmp_path: Path) -> None:
    prompt = build_prompt_records([paired_state(states)])[0]
    labels = tmp_path / "labels.jsonl"; output = tmp_path / "preferences.jsonl"; failed = tmp_path / "failed.jsonl"
    write_jsonl(labels, [
        {"prompt_id": prompt["prompt_id"], "chosen": prompt["action_a"], "rejected": prompt["action_b"], "confidence": 0.8, "reason": "valid"},
        {"prompt_id": prompt["prompt_id"], "chosen": "invalid", "rejected": prompt["action_b"], "confidence": 0.8, "reason": "bad"},
    ])
    result = run_replay([prompt], labels, output, failed)
    assert result == {"prompts": 1, "preferences": 1, "failed": 1}
    assert read_jsonl(output)[0]["usable_for_training"] is True
    assert read_jsonl(failed)[0]["validation_status"] == "invalid"


def test_offline_mode_does_not_create_fake_labels(states: list[dict[str, object]], tmp_path: Path) -> None:
    output = tmp_path / "preferences.jsonl"; output.write_text("stale\n", encoding="utf-8")
    result = run_offline(build_prompt_records([paired_state(states)]), output)
    assert result["preferences"] == 0
    assert not output.exists()


def test_api_invalid_response_is_saved(states: list[dict[str, object]], tmp_path: Path) -> None:
    prompt = build_prompt_records([paired_state(states)])[0]
    output = tmp_path / "preferences.jsonl"
    failed = tmp_path / "failed.jsonl"
    result = run_api([prompt], output, failed, "test-model", max_retries=1,
                     api_call=lambda _text, _model, _temperature: "not-json")
    assert result == {"prompts": 1, "preferences": 0, "failed": 1}
    assert read_jsonl(failed)[0]["parser_status"] == "error"
