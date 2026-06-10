"""Stage 4 RLAIF data workflow regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlaif.ai_evaluator import (
    APISettings,
    MISSING_API_MESSAGE,
    NoPreferenceLabelsError,
    load_api_settings,
    parse_ai_response,
    run_api,
    run_offline,
    run_replay,
)
from rlaif.collect_assignment_states import collect_assignment_states
from rlaif.preference_dataset import (
    CANDIDATE_FEATURE_KEYS,
    read_jsonl,
    validate_assignment_state,
    validate_prompt,
    write_jsonl,
)
from rlaif.prompt_builder import build_prompt_records, select_action_pairs
from experiments.build_ai_preferences import main as labeling_main

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


def test_candidate_action_features_are_context_not_labels(states: list[dict[str, object]]) -> None:
    for features in states[0]["candidate_action_features"].values():
        assert CANDIDATE_FEATURE_KEYS <= features.keys()
        assert isinstance(features["infeasibility_reasons"], list)
        assert "preference_score" not in features
        assert "chosen" not in features
        assert "rejected" not in features


def test_prompt_json_requirement_check(states: list[dict[str, object]]) -> None:
    prompt = build_prompt_records([paired_state(states)])[0]
    validate_prompt(prompt)
    assert "Return only valid JSON" in prompt["prompt_text"]
    assert '"chosen"' in prompt["prompt_text"]


def test_pair_selection_logic_does_not_create_labels(states: list[dict[str, object]]) -> None:
    state = paired_state(states)
    pairs = select_action_pairs(state)
    assert 1 <= len(pairs) <= 3
    assert any("TD" in pair for pair in pairs)
    assert len(pairs) == len(set(pairs))
    prompts = build_prompt_records([state])
    assert all("chosen" not in prompt and "rejected" not in prompt for prompt in prompts)


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


def test_replay_validates_user_labels_without_generating_more(states: list[dict[str, object]], tmp_path: Path) -> None:
    prompts = build_prompt_records([paired_state(states)])[:2]
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "preferences.jsonl"
    failed = tmp_path / "failed.jsonl"
    write_jsonl(labels, [{
        "prompt_id": prompts[0]["prompt_id"],
        "chosen": prompts[0]["action_a"],
        "rejected": prompts[0]["action_b"],
        "confidence": 0.8,
        "reason": "user supplied",
    }])
    result = run_replay(prompts, labels, output, failed)
    records = read_jsonl(output)
    assert result == {"prompts": len(prompts), "preferences": 1, "failed": 0}
    assert len(records) == 1
    assert records[0]["label_source"] == "user_provided_replay"
    assert records[0]["usable_for_training"] is True


def test_invalid_replay_labels_are_written_to_failed(states: list[dict[str, object]], tmp_path: Path) -> None:
    prompt = build_prompt_records([paired_state(states)])[0]
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "preferences.jsonl"
    failed = tmp_path / "failed.jsonl"
    write_jsonl(labels, [
        {"prompt_id": prompt["prompt_id"], "chosen": prompt["action_a"], "rejected": prompt["action_b"], "confidence": 0.8, "reason": "valid"},
        {"prompt_id": prompt["prompt_id"], "chosen": "invalid", "rejected": prompt["action_b"], "confidence": 0.8, "reason": "bad"},
    ])
    result = run_replay([prompt], labels, output, failed)
    assert result == {"prompts": 1, "preferences": 1, "failed": 1}
    assert read_jsonl(failed)[0]["validation_status"] == "invalid"


def test_all_invalid_replay_refuses_preference_file(states: list[dict[str, object]], tmp_path: Path) -> None:
    prompt = build_prompt_records([paired_state(states)])[0]
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "preferences.jsonl"
    failed = tmp_path / "failed.jsonl"
    output.write_text("stale\n", encoding="utf-8")
    write_jsonl(labels, [{"prompt_id": prompt["prompt_id"], "chosen": "invalid", "rejected": prompt["action_b"], "confidence": 0.8}])
    with pytest.raises(NoPreferenceLabelsError, match="Refusing"):
        run_replay([prompt], labels, output, failed)
    assert not output.exists()
    assert len(read_jsonl(failed)) == 1


def test_low_confidence_replay_is_valid_but_unusable(states: list[dict[str, object]], tmp_path: Path) -> None:
    prompt = build_prompt_records([paired_state(states)])[0]
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "preferences.jsonl"
    failed = tmp_path / "failed.jsonl"
    write_jsonl(labels, [{"prompt_id": prompt["prompt_id"], "chosen": prompt["action_a"], "rejected": prompt["action_b"], "confidence": 0.2, "reason": "uncertain"}])
    run_replay([prompt], labels, output, failed)
    assert read_jsonl(output)[0]["usable_for_training"] is False


def test_offline_creates_only_blank_templates(states: list[dict[str, object]], tmp_path: Path) -> None:
    prompts = build_prompt_records([paired_state(states)])
    output = tmp_path / "preferences.jsonl"
    template = tmp_path / "manual_labels_template.jsonl"
    small = tmp_path / "manual_labels_template_small.jsonl"
    output.write_text("stale\n", encoding="utf-8")
    result = run_offline(prompts, output, template, small, small_template_size=1)
    assert result["preferences"] == 0
    assert not output.exists()
    assert len(read_jsonl(template)) == len(prompts)
    assert len(read_jsonl(small)) == 1
    assert all(
        record["chosen"] is None
        and record["rejected"] is None
        and record["confidence"] is None
        and record["reason"] is None
        for record in read_jsonl(template)
    )


def test_api_without_configuration_does_not_create_labels(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("RLAIF_API_KEY", "RLAIF_API_BASE_URL", "RLAIF_MODEL_NAME"):
        monkeypatch.delenv(name, raising=False)
    output = tmp_path / "preferences.jsonl"
    output.write_text("stale\n", encoding="utf-8")
    with pytest.raises(NoPreferenceLabelsError, match="API mode is not configured"):
        load_api_settings()
    with pytest.raises(NoPreferenceLabelsError, match="API mode is not configured"):
        run_api([], output, tmp_path / "failed.jsonl")
    assert not output.exists()
    assert "replay mode" in MISSING_API_MESSAGE


def test_api_cli_missing_config_removes_stale_output(
    states: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in ("RLAIF_API_KEY", "RLAIF_API_BASE_URL", "RLAIF_MODEL_NAME"):
        monkeypatch.delenv(name, raising=False)
    prompts_path = tmp_path / "prompts.jsonl"
    output = tmp_path / "preferences.jsonl"
    output.write_text("stale\n", encoding="utf-8")
    write_jsonl(prompts_path, build_prompt_records([paired_state(states)]))
    status = labeling_main([
        "--mode", "api", "--prompts", str(prompts_path), "--output", str(output)
    ])
    assert status == 2
    assert not output.exists()


def test_api_invalid_response_is_failed_and_not_a_label(states: list[dict[str, object]], tmp_path: Path) -> None:
    prompt = build_prompt_records([paired_state(states)])[0]
    output = tmp_path / "preferences.jsonl"
    failed = tmp_path / "failed.jsonl"
    settings = APISettings("placeholder", "https://example.invalid/v1", "test-model", max_retries=1)
    with pytest.raises(NoPreferenceLabelsError, match="Refusing"):
        run_api([prompt], output, failed, settings, api_call=lambda _text, _settings: "not-json")
    assert not output.exists()
    assert read_jsonl(failed)[0]["parser_status"] == "error"


def test_api_valid_response_has_external_provenance(states: list[dict[str, object]], tmp_path: Path) -> None:
    prompt = build_prompt_records([paired_state(states)])[0]
    output = tmp_path / "preferences.jsonl"
    settings = APISettings("placeholder", "https://example.invalid/v1", "test-model", max_retries=1)
    raw = json.dumps({"chosen": prompt["action_a"], "rejected": prompt["action_b"], "confidence": 0.9, "reason": "API output"})
    run_api([prompt], output, tmp_path / "failed.jsonl", settings, api_call=lambda _text, _settings: raw)
    assert read_jsonl(output)[0]["label_source"] == "external_evaluator"
