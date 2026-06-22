"""Training utilities for the Stage 5 pairwise assignment reward model."""

from __future__ import annotations

import csv
import math
import random
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from rlaif.preference_dataset import (PreferenceExample, load_action_mapping, load_preference_examples,
                                      split_preference_examples)
from rlaif.reward_model import AssignmentRewardModel, pairwise_preference_loss
from utils.config import load_config

EPSILON = 1e-6


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _stack(examples: list[PreferenceExample]) -> dict[str, Tensor]:
    if not examples:
        return {}
    return {
        "chosen_state": torch.tensor([x.chosen_state_features for x in examples], dtype=torch.float32),
        "chosen_action": torch.tensor([x.chosen_action_features for x in examples], dtype=torch.float32),
        "chosen_id": torch.tensor([x.chosen_action_id for x in examples], dtype=torch.long),
        "rejected_state": torch.tensor([x.rejected_state_features for x in examples], dtype=torch.float32),
        "rejected_action": torch.tensor([x.rejected_action_features for x in examples], dtype=torch.float32),
        "rejected_id": torch.tensor([x.rejected_action_id for x in examples], dtype=torch.long),
    }


def compute_feature_statistics(train_examples: list[PreferenceExample]) -> dict[str, Tensor]:
    tensors = _stack(train_examples)
    states = torch.cat((tensors["chosen_state"], tensors["rejected_state"]), dim=0)
    actions = torch.cat((tensors["chosen_action"], tensors["rejected_action"]), dim=0)
    return {
        "state_feature_mean": states.mean(dim=0),
        "state_feature_std": states.std(dim=0, unbiased=False),
        "action_feature_mean": actions.mean(dim=0),
        "action_feature_std": actions.std(dim=0, unbiased=False),
    }


def normalize_tensors(tensors: dict[str, Tensor], stats: dict[str, Tensor]) -> dict[str, Tensor]:
    if not tensors:
        return {}
    normalized = dict(tensors)
    for side in ("chosen", "rejected"):
        normalized[f"{side}_state"] = (
            tensors[f"{side}_state"] - stats["state_feature_mean"]
        ) / (stats["state_feature_std"] + EPSILON)
        normalized[f"{side}_action"] = (
            tensors[f"{side}_action"] - stats["action_feature_mean"]
        ) / (stats["action_feature_std"] + EPSILON)
    if not all(torch.isfinite(value).all() for value in normalized.values()):
        raise ValueError("normalized reward-model features contain NaN or inf")
    return normalized


def _dataset(tensors: dict[str, Tensor]) -> TensorDataset | None:
    if not tensors:
        return None
    return TensorDataset(*(tensors[key] for key in (
        "chosen_state", "chosen_action", "chosen_id", "rejected_state", "rejected_action", "rejected_id"
    )))


def evaluate_tensors(model: AssignmentRewardModel, tensors: dict[str, Tensor],
                     batch_size: int = 256) -> dict[str, float | int | None]:
    dataset = _dataset(tensors)
    if dataset is None or len(dataset) == 0:
        return {"count": 0, "loss": None, "preference_accuracy": None, "chosen_score_mean": None,
                "rejected_score_mean": None, "score_margin_mean": None}
    model.eval()
    chosen_scores: list[Tensor] = []
    rejected_scores: list[Tensor] = []
    with torch.no_grad():
        for chosen_state, chosen_action, chosen_id, rejected_state, rejected_action, rejected_id in DataLoader(
                dataset, batch_size=batch_size, shuffle=False):
            chosen_scores.append(model(chosen_state, chosen_action, chosen_id))
            rejected_scores.append(model(rejected_state, rejected_action, rejected_id))
    chosen = torch.cat(chosen_scores)
    rejected = torch.cat(rejected_scores)
    margin = chosen - rejected
    return {
        "count": len(dataset),
        "loss": float(pairwise_preference_loss(chosen, rejected)),
        "preference_accuracy": float((margin > 0).float().mean()),
        "chosen_score_mean": float(chosen.mean()),
        "rejected_score_mean": float(rejected.mean()),
        "score_margin_mean": float(margin.mean()),
    }



def save_checkpoint(path: str | Path, checkpoint: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def train_reward_model(config: dict[str, Any], preferences_path: str | Path | None = None,
                       assignment_states_path: str | Path | None = None) -> dict[str, Any]:
    data_config, train_config = config["data"], config["training"]
    output_config, model_config = config["output"], config["reward_model"]
    seed = int(train_config["seed"])
    set_seed(seed)
    resolved_states_path = assignment_states_path or data_config["assignment_states_path"]
    examples = load_preference_examples(
        preferences_path or data_config["preferences_path"],
        resolved_states_path,
        min_confidence=float(data_config.get("min_confidence", 0.6)),
        use_only_usable_for_training=bool(data_config.get("use_only_usable_for_training", True)),
    )
    if len(examples) < 30:
        warnings.warn(
            f"Only {len(examples)} usable preference labels are available; this run is for smoke/pipeline "
            "validation, not final reward-model quality.", RuntimeWarning, stacklevel=2
        )
    splits = split_preference_examples(
        examples, float(data_config["train_ratio"]), float(data_config["val_ratio"]),
        float(data_config["test_ratio"]), seed,
    )
    stats = compute_feature_statistics(splits.train)
    split_tensors = {
        "train": normalize_tensors(_stack(splits.train), stats),
        "validation": normalize_tensors(_stack(splits.validation), stats),
        "test": normalize_tensors(_stack(splits.test), stats),
    }
    state_dim = len(examples[0].chosen_state_features)
    action_dim = len(examples[0].chosen_action_features)
    action_mapping = load_action_mapping(resolved_states_path)
    num_actions = max(action_mapping.values()) + 1
    model = AssignmentRewardModel(
        state_dim, action_dim, num_actions, int(model_config["action_emb_dim"]),
        tuple(model_config["hidden_dims"]), float(model_config.get("dropout", 0.0)),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_config["lr"]),
                                  weight_decay=float(train_config["weight_decay"]))
    train_dataset = _dataset(split_tensors["train"])
    assert train_dataset is not None
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_dataset, batch_size=int(train_config["batch_size"]), shuffle=True,
                        generator=generator)
    history: list[dict[str, Any]] = []
    best_state: dict[str, Tensor] | None = None
    best_val_loss = math.inf
    stale_epochs = 0
    patience = int(train_config["early_stopping_patience"])
    min_delta = float(train_config["min_delta"])
    for epoch in range(1, int(train_config["epochs"]) + 1):
        model.train()
        for batch in loader:
            chosen_state, chosen_action, chosen_id, rejected_state, rejected_action, rejected_id = batch
            chosen = model(chosen_state, chosen_action, chosen_id)
            rejected = model(rejected_state, rejected_action, rejected_id)
            loss = pairwise_preference_loss(chosen, rejected)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_metrics = evaluate_tensors(model, split_tensors["train"], int(train_config["batch_size"]))
        val_metrics = evaluate_tensors(model, split_tensors["validation"], int(train_config["batch_size"]))
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                        **{f"val_{k}": v for k, v in val_metrics.items()}})
        monitor = val_metrics["loss"] if val_metrics["loss"] is not None else train_metrics["loss"]
        assert isinstance(monitor, float)
        if monitor < best_val_loss - min_delta:
            best_val_loss = monitor
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = {name: evaluate_tensors(model, tensors, int(train_config["batch_size"]))
               for name, tensors in split_tensors.items()}
    train_scores = []
    model.eval()
    with torch.no_grad():
        tensors = split_tensors["train"]
        train_scores.extend((model(tensors["chosen_state"], tensors["chosen_action"], tensors["chosen_id"]),
                             model(tensors["rejected_state"], tensors["rejected_action"], tensors["rejected_id"])))
    all_train_scores = torch.cat(train_scores)
    reward_mean = float(all_train_scores.mean())
    reward_std = float(all_train_scores.std(unbiased=False))
    checkpoint = {
        "model_state_dict": model.state_dict(), "config": config, "feature_schema_version": "v2",
        "state_feature_dim": state_dim, "action_feature_dim": action_dim, "num_actions": num_actions,
        "action_mapping": action_mapping,
        **{key: value.cpu() for key, value in stats.items()},
        "reward_mean": reward_mean, "reward_std": reward_std,
        "train_metrics": metrics["train"], "val_metrics": metrics["validation"],
        "test_metrics": metrics["test"],
        "split_sizes": {"train": len(splits.train), "validation": len(splits.validation), "test": len(splits.test)},
        "usable_preference_count": len(examples),
    }
    save_checkpoint(output_config["checkpoint_path"], checkpoint)
    log_path = Path(output_config["training_log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0])
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    return {"checkpoint": checkpoint, "history": history, "model": model, "splits": splits}


def train_from_config(config_path: str | Path, preferences_path: str | Path | None = None) -> dict[str, Any]:
    return train_reward_model(load_config(config_path), preferences_path=preferences_path)
