"""Checkpoint evaluation for the Stage 5 assignment reward model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from rlaif.preference_dataset import load_preference_examples, split_preference_examples
from rlaif.reward_model import AssignmentRewardModel
from rlaif.train_reward_model import _stack, evaluate_tensors, load_checkpoint, normalize_tensors
from utils.config import load_config


def evaluate_reward_model(config: dict[str, Any], checkpoint_path: str | Path,
                          preferences_path: str | Path | None = None) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    data_config = config["data"]
    examples = load_preference_examples(
        preferences_path or data_config["preferences_path"], data_config["assignment_states_path"],
        float(data_config.get("min_confidence", 0.6)),
        bool(data_config.get("use_only_usable_for_training", True)),
    )
    splits = split_preference_examples(examples, float(data_config["train_ratio"]),
                                       float(data_config["val_ratio"]), float(data_config["test_ratio"]),
                                       int(config["training"]["seed"]))
    model_config = checkpoint["config"]["reward_model"]
    model = AssignmentRewardModel(
        int(checkpoint["state_feature_dim"]), int(checkpoint["action_feature_dim"]),
        int(checkpoint["num_actions"]), int(model_config["action_emb_dim"]),
        tuple(model_config["hidden_dims"]), float(model_config.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    stats = {key: torch.as_tensor(checkpoint[key]) for key in (
        "state_feature_mean", "state_feature_std", "action_feature_mean", "action_feature_std"
    )}
    test_examples = splits.test or splits.validation or splits.train
    metrics = evaluate_tensors(model, normalize_tensors(_stack(test_examples), stats),
                               int(config["training"]["batch_size"]))
    result = {
        "checkpoint_path": str(checkpoint_path), "evaluation_split": (
            "test" if splits.test else "validation" if splits.validation else "train"
        ), "metrics": metrics, "reward_mean": checkpoint["reward_mean"],
        "reward_std": checkpoint["reward_std"], "usable_preference_count": len(examples),
        "split_sizes": {"train": len(splits.train), "validation": len(splits.validation), "test": len(splits.test)},
    }
    output = Path(config["output"]["eval_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def evaluate_from_config(config_path: str | Path, checkpoint_path: str | Path,
                         preferences_path: str | Path | None = None) -> dict[str, Any]:
    return evaluate_reward_model(load_config(config_path), checkpoint_path, preferences_path)
