"""Hardening tests for the shared Stage 3/4/5 assignment feature schema."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from data_pipeline.build_instance import build_instance
from envs import DynamicDeliveryEnv
from envs import state_builder
from rlaif.collect_assignment_states import collect_assignment_states
from rlaif.preference_dataset import load_preference_examples, write_jsonl
from rlaif.prompt_builder import select_action_pairs

CONFIG = Path(__file__).parents[1] / "configs/shanghai_small.yaml"


@pytest.fixture()
def environment(tmp_path: Path) -> DynamicDeliveryEnv:
    instance = build_instance(CONFIG, fallback=True, output_root=tmp_path)
    env = DynamicDeliveryEnv(Path(instance["output_directory"]) / "instance.json")
    env.reset()
    return env


@pytest.fixture(scope="module")
def assignment_states(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, object]]:
    output = tmp_path_factory.mktemp("expanded-assignment-schema") / "states.jsonl"
    return collect_assignment_states(CONFIG, 1, output, fallback=True)


def test_assignment_feature_length_is_17_plus_10_per_station(
    environment: DynamicDeliveryEnv,
) -> None:
    observation = environment._observation()

    assert len(observation["features"]) == 17 + 10 * len(environment.station_ids)


def test_assignment_feature_names_are_exported(environment: DynamicDeliveryEnv) -> None:
    helper = getattr(state_builder, "assignment_feature_names", None)

    assert callable(helper)
    names = helper(environment.station_ids)
    assert len(names) == 17 + 10 * len(environment.station_ids)
    assert names[:3] == ("time_norm", "deadline_remaining_norm", "weight_norm")


def test_assignment_features_are_finite_numeric_values(
    environment: DynamicDeliveryEnv,
) -> None:
    features = environment._observation()["features"]

    assert all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in features)


def test_station_feature_blocks_follow_sorted_station_id_order(
    environment: DynamicDeliveryEnv,
) -> None:
    parcel = environment.parcels[environment.current_decision.event.payload["parcel_id"]]
    sorted_ids = sorted(environment.station_ids)
    for index, station_id in enumerate(sorted_ids, start=1):
        station = environment.stations[station_id]
        station.locker_load_kg = station.locker_capacity_kg * (index / 10.0)
    environment.station_ids = list(reversed(environment.station_ids))

    features = state_builder.build_assignment_features(environment, parcel)
    station_names = getattr(state_builder, "ASSIGNMENT_STATION_FEATURE_NAMES", ())
    occupancy_offset = station_names.index("locker_occupancy_ratio")
    observed = [
        features[17 + block * 10 + occupancy_offset]
        for block in range(len(sorted_ids))
    ]

    assert observed == pytest.approx([index / 10.0 for index in range(1, len(sorted_ids) + 1)])


def test_stage4_states_export_the_expanded_assignment_schema(
    assignment_states: list[dict[str, object]],
) -> None:
    state = assignment_states[0]
    expected_names = state_builder.assignment_feature_names(
        [row["station_id"] for row in state["station_states"]]
    )

    assert tuple(state["assignment_feature_names"]) == expected_names
    assert len(state["assignment_features"]) == len(expected_names)


def test_reward_model_dataset_loads_expanded_assignment_states(
    assignment_states: list[dict[str, object]], tmp_path: Path
) -> None:
    state = next(item for item in assignment_states if select_action_pairs(item))
    action_a, action_b = select_action_pairs(state)[0]
    states_path = tmp_path / "states.jsonl"
    preferences_path = tmp_path / "preferences.jsonl"
    write_jsonl(states_path, [state])
    write_jsonl(preferences_path, [{
        "preference_id": "expanded-schema-test",
        "state_id": state["state_id"],
        "action_a": action_a,
        "action_b": action_b,
        "chosen": action_a,
        "rejected": action_b,
        "confidence": 1.0,
        "validation_status": "valid",
        "usable_for_training": True,
        "label_source": "test_fixture",
    }])

    examples = load_preference_examples(preferences_path, states_path)

    assert len(examples) == 1
    assert len(examples[0].chosen_state_features) == 17 + 10 * len(state["station_states"])
