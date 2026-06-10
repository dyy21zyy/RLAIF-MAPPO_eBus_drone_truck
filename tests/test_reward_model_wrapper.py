"""Strict learned-reward wrapper behavior tests."""

from pathlib import Path

import pytest

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
    torch = pytest.importorskip("torch")
    path = tmp_path / "invalid.pt"
    torch.save({"not": "a reward model"}, path)
    with pytest.raises(RewardModelCheckpointError, match="Invalid Stage 5"):
        RewardModelWrapper(path, enabled=True)
