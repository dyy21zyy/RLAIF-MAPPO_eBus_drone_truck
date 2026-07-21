from __future__ import annotations

import math
from pathlib import Path

import pytest

from tests.torch_optional import import_optional_torch

torch = import_optional_torch(allow_module_level=True)

from experiments.smoke_test_reward_model import _config, build_fixture
from rlaif.preference_dataset import load_preference_examples, split_preference_examples
from rlaif.reward_model import AssignmentRewardModel, normalize_reward, pairwise_preference_loss
from rlaif.train_reward_model import (_stack, compute_feature_statistics, load_checkpoint,
                                      normalize_tensors, train_reward_model)


def test_feature_normalization_has_no_nan_or_inf(tmp_path: Path) -> None:
    states, preferences = build_fixture(tmp_path)
    split = split_preference_examples(load_preference_examples(preferences, states)).train
    normalized = normalize_tensors(_stack(split), compute_feature_statistics(split))
    assert all(torch.isfinite(value).all() for value in normalized.values())


def test_model_forward_and_pairwise_loss() -> None:
    model = AssignmentRewardModel(6, 10, 3)
    chosen = model(torch.zeros(4, 6), torch.zeros(4, 10), torch.zeros(4, dtype=torch.long))
    rejected = model(torch.ones(4, 6), torch.ones(4, 10), torch.ones(4, dtype=torch.long))
    assert chosen.shape == (4,)
    assert torch.isfinite(pairwise_preference_loss(chosen, rejected))


def test_checkpoint_round_trip_and_reward_normalization(tmp_path: Path) -> None:
    states, preferences = build_fixture(tmp_path)
    config = _config(tmp_path, states, preferences)
    config["training"]["epochs"] = 1
    result = train_reward_model(config)
    loaded = load_checkpoint(config["output"]["checkpoint_path"])
    required = {"model_state_dict", "config", "feature_schema_version", "state_feature_dim",
                "action_feature_dim", "num_actions", "action_mapping", "state_feature_mean",
                "state_feature_std", "action_feature_mean", "action_feature_std", "reward_mean",
                "reward_std", "train_metrics", "val_metrics", "test_metrics"}
    assert required <= loaded.keys()
    assert math.isfinite(loaded["reward_mean"]) and math.isfinite(loaded["reward_std"])
    normalized = normalize_reward(torch.tensor([loaded["reward_mean"]]), loaded["reward_mean"],
                                  loaded["reward_std"])
    assert torch.isfinite(normalized).all()
    assert Path(config["output"]["checkpoint_path"]).is_file()
