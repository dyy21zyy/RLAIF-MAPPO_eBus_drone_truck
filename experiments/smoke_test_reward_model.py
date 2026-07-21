"""Offline Stage 5 reward-model smoke test using temporary labeled fixtures."""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

from rlaif.preference_dataset import (ACTION_FEATURE_KEYS, NO_USABLE_LABELS_MESSAGE,
                                      NoUsablePreferencesError, write_jsonl)
from rlaif.torch_runtime import PYTORCH_REQUIRED_MESSAGE, is_missing_torch_error, is_torch_runtime_available


def _action(action_id: int, name: str, offset: float) -> dict[str, object]:
    values = {key: float(index + offset) for index, key in enumerate(ACTION_FEATURE_KEYS)}
    values["feasible_flag"] = 1.0
    return {"action_id": action_id, "action_name": name, **values, "infeasibility_reasons": []}


def build_fixture(directory: Path, count: int = 10) -> tuple[Path, Path]:
    states, preferences = [], []
    for index in range(count):
        state_id = f"smoke-state-{index}"
        states.append({
            "state_id": state_id, "feature_schema_version": "v2",
            "assignment_features": [index / 10, 0.1, 0.5, 1.0, 1.0, 0.0],
            "candidate_action_features": {
                "TD": _action(0, "TD", 0.2 + index / 100),
                "TBD_station_01": _action(1, "TBD_station_01", 1.2 + index / 100),
            },
        })
        preferences.append({
            "preference_id": f"smoke-pref-{index}", "state_id": state_id,
            "action_a": "TD", "action_b": "TBD_station_01", "chosen": "TD",
            "rejected": "TBD_station_01", "confidence": 0.9, "validation_status": "valid",
            "usable_for_training": True, "label_source": "temporary_smoke_fixture",
        })
    states_path, preferences_path = directory / "states.jsonl", directory / "preferences.jsonl"
    write_jsonl(states_path, states)
    write_jsonl(preferences_path, preferences)
    return states_path, preferences_path


def _config(directory: Path, states: Path, preferences: Path) -> dict[str, object]:
    return {
        "reward_model": {"action_emb_dim": 16, "hidden_dims": [32, 16], "dropout": 0.0},
        "training": {"batch_size": 8, "epochs": 2, "lr": 0.001, "weight_decay": 0.0,
                     "seed": 42, "early_stopping_patience": 2, "min_delta": 0.0},
        "data": {"preferences_path": str(preferences), "assignment_states_path": str(states),
                 "train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1,
                 "min_confidence": 0.6, "use_only_usable_for_training": True},
        "output": {"checkpoint_path": str(directory / "reward_model.pt"),
                   "training_log_path": str(directory / "training.csv"),
                   "eval_path": str(directory / "eval.json")},
    }


def main() -> int:
    if not is_torch_runtime_available():
        print(PYTORCH_REQUIRED_MESSAGE, file=sys.stderr)
        return 3
    try:
        import torch

        from rlaif.evaluate_reward_model import evaluate_reward_model
        from rlaif.reward_model import AssignmentRewardModel, normalize_reward
        from rlaif.train_reward_model import load_checkpoint, train_reward_model
    except ModuleNotFoundError as exc:
        if not is_missing_torch_error(exc):
            raise
        print(PYTORCH_REQUIRED_MESSAGE, file=sys.stderr)
        return 3

    with tempfile.TemporaryDirectory(prefix="stage5-smoke-") as temporary:
        directory = Path(temporary)
        states, preferences = build_fixture(directory)
        config = _config(directory, states, preferences)
        result = train_reward_model(config)  # type: ignore[arg-type]
        checkpoint = result["checkpoint"]
        assert math.isfinite(checkpoint["train_metrics"]["loss"])
        assert checkpoint["test_metrics"]["preference_accuracy"] is not None
        checkpoint_path = Path(config["output"]["checkpoint_path"])  # type: ignore[index]
        assert checkpoint_path.is_file()
        loaded = load_checkpoint(checkpoint_path)
        model_config = loaded["config"]["reward_model"]
        model = AssignmentRewardModel(loaded["state_feature_dim"], loaded["action_feature_dim"],
                                      loaded["num_actions"], model_config["action_emb_dim"],
                                      model_config["hidden_dims"], model_config["dropout"])
        model.load_state_dict(loaded["model_state_dict"])
        score = model(torch.zeros(1, loaded["state_feature_dim"]),
                      torch.zeros(1, loaded["action_feature_dim"]), torch.zeros(1, dtype=torch.long))
        assert score.shape == (1,) and torch.isfinite(score).all()
        assert torch.isfinite(normalize_reward(score, loaded["reward_mean"], loaded["reward_std"])).all()
        evaluation = evaluate_reward_model(config, checkpoint_path)  # type: ignore[arg-type]
        assert math.isfinite(evaluation["metrics"]["loss"])
        empty = directory / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        for invalid_path in (directory / "missing.jsonl", empty):
            bad_config = dict(config)
            bad_config["data"] = {**config["data"], "preferences_path": str(invalid_path)}  # type: ignore[arg-type]
            try:
                train_reward_model(bad_config)  # type: ignore[arg-type]
            except NoUsablePreferencesError as exc:
                assert str(exc) == NO_USABLE_LABELS_MESSAGE
            else:
                raise AssertionError("missing/empty labels did not fail gracefully")
        assert all(record.get("label_source") != "rule_based" for record in (
            json.loads(line) for line in preferences.read_text(encoding="utf-8").splitlines()
        ))
    print("Stage 5 reward-model smoke test passed (offline, no external API).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
