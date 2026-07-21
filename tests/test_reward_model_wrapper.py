"""Strict learned-reward wrapper behavior tests."""

from pathlib import Path

import pytest

from tests.torch_optional import import_optional_torch
from training.reward_model_wrapper import RewardModelCheckpointError, RewardModelWrapper


def test_disabled_rlaif_does_not_require_checkpoint(tmp_path: Path) -> None:
    wrapper = RewardModelWrapper(tmp_path / "missing.pt", enabled=False)
    assert wrapper.enabled is False
    with pytest.raises(RuntimeError, match="no substitute reward"):
        wrapper.score([0.0], [0.0], 0)


def test_enabled_rlaif_requires_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(RewardModelCheckpointError, match="requires a valid Stage 5"):
        RewardModelWrapper(tmp_path / "missing.pt", enabled=True)


def test_invalid_checkpoint_does_not_fabricate_reward(tmp_path: Path) -> None:
    torch = import_optional_torch()
    path = tmp_path / "invalid.pt"
    torch.save({"not": "a reward model"}, path)
    with pytest.raises(RewardModelCheckpointError, match="Invalid Stage 5"):
        RewardModelWrapper(path, enabled=True)


def _constant_checkpoint(tmp_path: Path, bias: float = 0.0) -> Path:
    torch = import_optional_torch()
    from rlaif.reward_model import AssignmentRewardModel
    model = AssignmentRewardModel(1, 1, 2, hidden_dims=[], action_emb_dim=1)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.mlp[-1].bias.fill_(bias)
    path = tmp_path / f"constant_{bias}.pt"
    torch.save({
        "model_state_dict": model.state_dict(), "state_feature_dim": 1, "action_feature_dim": 1, "num_actions": 2,
        "state_feature_mean": torch.tensor([0.0]), "state_feature_std": torch.tensor([1.0]),
        "action_feature_mean": torch.tensor([0.0]), "action_feature_std": torch.tensor([1.0]),
        "reward_mean": 0.0, "reward_std": 1.0, "config": {"reward_model": {"hidden_dims": [], "action_emb_dim": 1}},
    }, path)
    return path


def test_missing_model_falls_back_when_configured(tmp_path: Path) -> None:
    wrapper = RewardModelWrapper(tmp_path / "missing.pt", enabled=True, fallback_to_env_reward=True, fail_on_invalid_reward_model=False)
    assert wrapper.fallen_back is True
    assert wrapper.enabled is False


def test_invalid_model_raises_when_fallback_disabled(tmp_path: Path) -> None:
    torch = import_optional_torch()
    path = tmp_path / "bad.pt"
    torch.save({"bad": "checkpoint"}, path)
    with pytest.raises(RewardModelCheckpointError, match="Invalid Stage 5"):
        RewardModelWrapper(path, enabled=True, fallback_to_env_reward=False, fail_on_invalid_reward_model=True)


def test_validation_failure_can_fall_back(tmp_path: Path) -> None:
    ckpt = _constant_checkpoint(tmp_path)
    prefs = tmp_path / "prefs.jsonl"
    prefs.write_text('{"state_features":[0],"chosen_action_features":[0],"chosen_action_id":0,"rejected_action_features":[0],"rejected_action_id":1}\n')
    wrapper = RewardModelWrapper(ckpt, enabled=True, validation={"enabled": True, "preference_data": str(prefs), "min_pairwise_accuracy": 1.0, "min_samples": 1}, fallback_to_env_reward=True, fail_on_invalid_reward_model=False)
    assert wrapper.fallen_back is True


def test_reward_clipping_after_standardization(tmp_path: Path) -> None:
    positive = RewardModelWrapper(_constant_checkpoint(tmp_path, 9.0), enabled=True, reward_clip=5.0)
    negative = RewardModelWrapper(_constant_checkpoint(tmp_path, -9.0), enabled=True, reward_clip=5.0)
    unclipped = RewardModelWrapper(_constant_checkpoint(tmp_path, 9.0), enabled=True, reward_clip=None)
    assert positive.score([0], [0], 0) == 5.0
    assert negative.score([0], [0], 0) == -5.0
    assert unclipped.score([0], [0], 0) > 5.0
