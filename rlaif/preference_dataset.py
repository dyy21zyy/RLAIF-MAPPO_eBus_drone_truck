"""Stage 4 JSONL validation and Stage 5 pairwise reward-model datasets."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

NO_USABLE_LABELS_MESSAGE = (
    "No usable AI/human preference labels found. Run Stage 4 in API mode or replay mode "
    "with valid labels before training the reward model."
)
ACTION_FEATURE_KEYS = (
    "feasible_flag",
    "estimated_delivery_time",
    "estimated_lateness",
    "estimated_truck_distance",
    "estimated_truck_time",
    "estimated_bus_wait_time",
    "estimated_bus_linehaul_time",
    "estimated_drone_time",
    "estimated_locker_load_after_assignment",
    "estimated_station_power_margin",
)
CANDIDATE_FEATURE_KEYS = {"action_id", "action_name", *ACTION_FEATURE_KEYS, "infeasibility_reasons"}


class NoUsablePreferencesError(ValueError):
    """Raised when Stage 5 has no approved labels to train from."""

    def __init__(self) -> None:
        super().__init__(NO_USABLE_LABELS_MESSAGE)


@dataclass(frozen=True)
class PreferenceExample:
    preference_id: str
    state_id: str
    chosen_action_name: str
    chosen_state_features: tuple[float, ...]
    chosen_action_features: tuple[float, ...]
    chosen_action_id: int
    rejected_action_name: str
    rejected_state_features: tuple[float, ...]
    rejected_action_features: tuple[float, ...]
    rejected_action_id: int


@dataclass(frozen=True)
class DatasetSplits:
    train: list[PreferenceExample]
    validation: list[PreferenceExample]
    test: list[PreferenceExample]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} is not an object")
            records.append(record)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    return count


def validate_assignment_state(record: dict[str, Any]) -> None:
    required = {"state_id", "episode_id", "current_time", "feature_schema_version", "assignment_features",
                "assignment_feature_names",
                "parcel", "system_state_summary", "station_states", "candidate_actions",
                "candidate_action_features", "action_mask"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"assignment state missing keys: {sorted(missing)}")
    if record["feature_schema_version"] != "v1" or not record["assignment_features"]:
        raise ValueError("invalid assignment feature schema")
    if len(record["assignment_features"]) != len(record["assignment_feature_names"]):
        raise ValueError("assignment feature values and names differ in length")
    if len(record["candidate_actions"]) != len(record["action_mask"]):
        raise ValueError("candidate actions and action mask differ in length")
    for action in record["candidate_actions"]:
        features = record["candidate_action_features"].get(action["action_name"])
        if not isinstance(features, dict) or CANDIDATE_FEATURE_KEYS - features.keys():
            raise ValueError(f"invalid candidate features for {action['action_name']}")
        if features["feasible_flag"] != action["feasible"]:
            raise ValueError("candidate feasibility fields disagree")
        if not action["feasible"] and not features["infeasibility_reasons"]:
            raise ValueError("infeasible candidate lacks reasons")


def validate_prompt(record: dict[str, Any]) -> None:
    required = {"prompt_id", "state_id", "action_a", "action_b", "prompt_version", "prompt_text", "metadata"}
    if required - record.keys() or record["action_a"] == record["action_b"]:
        raise ValueError("invalid prompt record")
    text = record["prompt_text"]
    for phrase in ("Return only valid JSON", "chosen must be either action_a or action_b", '"confidence"'):
        if phrase not in text:
            raise ValueError(f"prompt missing requirement: {phrase}")


def validate_preference(record: dict[str, Any], confidence_threshold: float = 0.6) -> dict[str, Any]:
    action_a, action_b = record["action_a"], record["action_b"]
    if record.get("chosen") not in {action_a, action_b}:
        raise ValueError("chosen is not one of the compared actions")
    expected_rejected = action_b if record["chosen"] == action_a else action_a
    if record.get("rejected") != expected_rejected:
        raise ValueError("rejected is not the other compared action")
    confidence = float(record["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    record["confidence"] = confidence
    record["parser_status"] = "ok"
    record["validation_status"] = "valid"
    record["usable_for_training"] = confidence >= confidence_threshold
    return record


def _finite_vector(values: Any, name: str) -> tuple[float, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} contains NaN or inf")
    return result


def load_preference_examples(preferences_path: str | Path, assignment_states_path: str | Path,
                             min_confidence: float = 0.6,
                             use_only_usable_for_training: bool = True) -> list[PreferenceExample]:
    """Join approved preferences to Stage 4 states without creating or repairing labels."""
    try:
        preferences = read_jsonl(preferences_path)
    except FileNotFoundError as exc:
        raise NoUsablePreferencesError() from exc
    if not preferences:
        raise NoUsablePreferencesError()
    try:
        states = read_jsonl(assignment_states_path)
    except FileNotFoundError as exc:
        raise ValueError(f"Assignment-state file not found: {assignment_states_path}") from exc
    state_by_id = {str(state.get("state_id")): state for state in states}
    examples: list[PreferenceExample] = []
    expected_state_dim: int | None = None
    expected_action_dim: int | None = None
    for record in preferences:
        if record.get("validation_status") != "valid":
            continue
        if use_only_usable_for_training and record.get("usable_for_training") is not True:
            continue
        try:
            confidence = float(record.get("confidence", -1.0))
        except (TypeError, ValueError):
            continue
        if confidence < min_confidence:
            continue
        chosen, rejected = record.get("chosen"), record.get("rejected")
        action_a, action_b = record.get("action_a"), record.get("action_b")
        if chosen not in {action_a, action_b} or rejected not in {action_a, action_b} or chosen == rejected:
            continue
        state = state_by_id.get(str(record.get("state_id")))
        if state is None or state.get("feature_schema_version") != "v1":
            continue
        candidates = state.get("candidate_action_features", {})
        if chosen not in candidates or rejected not in candidates:
            continue
        try:
            state_features = _finite_vector(state.get("assignment_features"), "assignment_features")
            chosen_raw, rejected_raw = candidates[chosen], candidates[rejected]
            chosen_features = _finite_vector(
                [float(chosen_raw[key]) for key in ACTION_FEATURE_KEYS], "chosen action features"
            )
            rejected_features = _finite_vector(
                [float(rejected_raw[key]) for key in ACTION_FEATURE_KEYS], "rejected action features"
            )
            chosen_id, rejected_id = int(chosen_raw["action_id"]), int(rejected_raw["action_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if chosen_id < 0 or rejected_id < 0:
            continue
        if expected_state_dim is None:
            expected_state_dim, expected_action_dim = len(state_features), len(chosen_features)
        if len(state_features) != expected_state_dim or len(chosen_features) != expected_action_dim \
                or len(rejected_features) != expected_action_dim:
            continue
        examples.append(PreferenceExample(
            preference_id=str(record.get("preference_id", "")), state_id=str(record["state_id"]),
            chosen_action_name=str(chosen), chosen_state_features=state_features, chosen_action_features=chosen_features,
            chosen_action_id=chosen_id, rejected_action_name=str(rejected), rejected_state_features=state_features,
            rejected_action_features=rejected_features, rejected_action_id=rejected_id,
        ))
    if not examples:
        raise NoUsablePreferencesError()
    return examples


def load_action_mapping(assignment_states_path: str | Path) -> dict[str, int]:
    """Load the complete stable Stage 4 candidate-name to action-ID mapping."""
    states = read_jsonl(assignment_states_path)
    mapping: dict[str, int] = {}
    for state in states:
        for name, features in state.get("candidate_action_features", {}).items():
            try:
                action_id = int(features["action_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if name in mapping and mapping[name] != action_id:
                raise ValueError(f"inconsistent action ID for {name}")
            mapping[str(name)] = action_id
    if not mapping or sorted(mapping.values()) != list(range(max(mapping.values()) + 1)):
        raise ValueError("assignment states do not contain a contiguous candidate action mapping")
    return dict(sorted(mapping.items(), key=lambda item: item[1]))


def split_preference_examples(examples: Sequence[PreferenceExample], train_ratio: float = 0.8,
                              val_ratio: float = 0.1, test_ratio: float = 0.1,
                              seed: int = 42) -> DatasetSplits:
    if not examples:
        raise NoUsablePreferencesError()
    if min(train_ratio, val_ratio, test_ratio) < 0 or not math.isclose(
            train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-6):
        raise ValueError("train/validation/test ratios must be non-negative and sum to 1")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    if count >= 3:
        val_count = max(1, int(count * val_ratio))
        test_count = max(1, int(count * test_ratio))
        train_count = count - val_count - test_count
        if train_count < 1:
            train_count, val_count, test_count = count - 2, 1, 1
    else:
        train_count, val_count = 1, 1 if count == 2 else 0
        test_count = count - train_count - val_count
    return DatasetSplits(
        train=shuffled[:train_count],
        validation=shuffled[train_count:train_count + val_count],
        test=shuffled[train_count + val_count:],
    )
