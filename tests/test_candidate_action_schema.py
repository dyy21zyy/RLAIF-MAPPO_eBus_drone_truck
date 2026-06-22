"""Stable candidate-action schema tests shared by Stages 3, 4, and 5."""

from __future__ import annotations

from pathlib import Path

import pytest

from data_pipeline.build_instance import build_instance
from envs import DynamicDeliveryEnv
from envs import state_builder
from experiments.smoke_test_reward_model import build_fixture
from rlaif.collect_assignment_states import collect_assignment_states
from rlaif.preference_dataset import ACTION_FEATURE_KEYS, load_preference_examples
from rlaif.prompt_builder import build_prompt_records, select_action_pairs

CONFIG = Path(__file__).parents[1] / "configs/shanghai_small.yaml"


@pytest.fixture()
def environment(tmp_path: Path) -> DynamicDeliveryEnv:
    instance = build_instance(CONFIG, fallback=True, output_root=tmp_path)
    env = DynamicDeliveryEnv(Path(instance["output_directory"]) / "instance.json")
    env.reset()
    return env


def _features(environment: DynamicDeliveryEnv, action_id: int) -> dict[str, object]:
    parcel = environment.parcels[environment.current_decision.event.payload["parcel_id"]]
    return state_builder.build_candidate_action_features(environment, parcel, action_id, False)


def test_candidate_action_numeric_vector_length_is_stable(
    environment: DynamicDeliveryEnv,
) -> None:
    keys = getattr(state_builder, "CANDIDATE_ACTION_FEATURE_NAMES", ())
    candidate = _features(environment, 0)

    assert len(keys) == 14
    assert len([float(candidate[key]) for key in keys]) == 14


def test_candidate_action_type_one_hot_fields_are_correct(
    environment: DynamicDeliveryEnv,
) -> None:
    station_count = len(environment.station_ids)

    assert [_features(environment, 0)[key] for key in (
        "action_type_TD", "action_type_TBD", "action_type_TLD"
    )] == [1.0, 0.0, 0.0]
    assert [_features(environment, 1)[key] for key in (
        "action_type_TD", "action_type_TBD", "action_type_TLD"
    )] == [0.0, 1.0, 0.0]
    assert [_features(environment, station_count + 1)[key] for key in (
        "action_type_TD", "action_type_TBD", "action_type_TLD"
    )] == [0.0, 0.0, 1.0]


def test_candidate_station_index_uses_one_based_normalization(
    environment: DynamicDeliveryEnv,
) -> None:
    station_count = len(environment.station_ids)

    assert _features(environment, 0)["action_station_index_norm"] == 0.0
    assert _features(environment, 1)["action_station_index_norm"] == pytest.approx(1 / station_count)
    assert _features(environment, station_count)["action_station_index_norm"] == 1.0
    assert _features(environment, 2 * station_count)["action_station_index_norm"] == 1.0


def test_infeasible_candidate_has_human_readable_reasons(
    environment: DynamicDeliveryEnv,
) -> None:
    candidate = _features(environment, 1)

    assert candidate["feasible_flag"] == 0.0
    assert candidate["infeasibility_reasons"]


def test_reward_dataset_uses_shared_ordered_candidate_keys(tmp_path: Path) -> None:
    states, preferences = build_fixture(tmp_path, count=1)
    examples = load_preference_examples(preferences, states)
    state = __import__("json").loads(states.read_text(encoding="utf-8").splitlines()[0])
    expected = tuple(float(state["candidate_action_features"]["TD"][key]) for key in ACTION_FEATURE_KEYS)

    assert ACTION_FEATURE_KEYS == state_builder.CANDIDATE_ACTION_FEATURE_NAMES
    assert examples[0].chosen_action_features == expected


def test_prompt_contains_objective_action_context_but_no_label(
    tmp_path: Path,
) -> None:
    states = collect_assignment_states(CONFIG, 1, tmp_path / "states.jsonl", fallback=True)
    state = next(item for item in states if select_action_pairs(item))
    prompt = build_prompt_records([state])[0]

    assert "action_type_TD" in prompt["prompt_text"]
    assert "estimated_delivery_time_norm" in prompt["prompt_text"]
    assert "infeasibility_reasons" in prompt["prompt_text"]
    assert "chosen" not in prompt
    assert "rejected" not in prompt
