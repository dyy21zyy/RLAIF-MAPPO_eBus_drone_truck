from __future__ import annotations

from pathlib import Path

import pytest

from experiments.smoke_test_reward_model import build_fixture
from rlaif.preference_dataset import (NO_USABLE_LABELS_MESSAGE, NoUsablePreferencesError,
                                      load_preference_examples, split_preference_examples, write_jsonl)


def test_preference_dataset_loading_and_tiny_split(tmp_path: Path) -> None:
    states, preferences = build_fixture(tmp_path)
    examples = load_preference_examples(preferences, states)
    splits = split_preference_examples(examples, seed=42)
    assert len(examples) == 10
    assert (len(splits.train), len(splits.validation), len(splits.test)) == (8, 1, 1)
    assert examples[0].chosen_action_id == 0
    assert examples[0].rejected_action_id == 1


def test_filters_unusable_and_invalid_choices(tmp_path: Path) -> None:
    states, preferences = build_fixture(tmp_path, count=3)
    records = [__import__("json").loads(line) for line in preferences.read_text().splitlines()]
    records[0]["usable_for_training"] = False
    records[1]["chosen"] = records[1]["rejected"]
    write_jsonl(preferences, records)
    examples = load_preference_examples(preferences, states)
    assert [example.preference_id for example in examples] == ["smoke-pref-2"]


@pytest.mark.parametrize("kind", ["missing", "empty"])
def test_missing_or_empty_preferences_fail_gracefully(tmp_path: Path, kind: str) -> None:
    states, _ = build_fixture(tmp_path)
    path = tmp_path / f"{kind}.jsonl"
    if kind == "empty":
        path.write_text("")
    with pytest.raises(NoUsablePreferencesError, match="No usable AI/human") as exc:
        load_preference_examples(path, states)
    assert str(exc.value) == NO_USABLE_LABELS_MESSAGE


def test_all_invalid_chosen_rejected_fails_gracefully(tmp_path: Path) -> None:
    states, preferences = build_fixture(tmp_path, count=1)
    record = __import__("json").loads(preferences.read_text())
    record["chosen"] = "unknown"
    write_jsonl(preferences, [record])
    with pytest.raises(NoUsablePreferencesError):
        load_preference_examples(preferences, states)
