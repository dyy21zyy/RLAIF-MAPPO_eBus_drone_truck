"""Strict Stage 5 learned-reward checkpoint adapter for Stage 6 PPO and Stage 7 MAPPO."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Sequence


class RewardModelCheckpointError(RuntimeError):
    """Raised when learned RLAIF reward was requested but cannot be loaded safely."""


class RewardModelWrapper:
    """Score normalized objective features; disabled mode never requires a checkpoint."""

    REQUIRED_KEYS = {
        "model_state_dict", "state_feature_dim", "action_feature_dim", "num_actions",
        "state_feature_mean", "state_feature_std", "action_feature_mean", "action_feature_std",
        "reward_mean", "reward_std",
    }

    def __init__(self, checkpoint_path: str | Path | None, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model = None
        self.checkpoint = None
        if not self.enabled:
            return
        if self.checkpoint_path is None or not self.checkpoint_path.is_file():
            raise RewardModelCheckpointError(
                f"rlaif_enabled=true requires a valid Stage 5 reward model checkpoint: {self.checkpoint_path}"
            )
        try:
            torch = importlib.import_module("torch")
            from rlaif.reward_model import AssignmentRewardModel

            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, dict) or self.REQUIRED_KEYS - checkpoint.keys():
                missing = sorted(self.REQUIRED_KEYS - set(checkpoint) if isinstance(checkpoint, dict) else self.REQUIRED_KEYS)
                raise ValueError(f"missing checkpoint keys: {missing}")
            model_config = checkpoint.get("config", {}).get("reward_model", {})
            model = AssignmentRewardModel(
                state_dim=int(checkpoint["state_feature_dim"]),
                action_feature_dim=int(checkpoint["action_feature_dim"]),
                num_actions=int(checkpoint["num_actions"]),
                hidden_dims=model_config.get("hidden_dims", [256, 256, 128]),
                action_emb_dim=int(model_config.get("action_emb_dim", 16)),
                dropout=float(model_config.get("dropout", 0.0)),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
        except Exception as exc:
            raise RewardModelCheckpointError(
                f"Invalid Stage 5 reward model checkpoint {self.checkpoint_path}: {exc}"
            ) from exc
        self.torch = torch
        self.checkpoint = checkpoint
        self.model = model

    def score(self, state_features: Sequence[float], action_features: Sequence[float], action_id: int) -> float:
        if not self.enabled or self.model is None or self.checkpoint is None:
            raise RuntimeError("RLAIF scoring is disabled; no substitute reward is available")
        torch = self.torch
        state = torch.tensor([state_features], dtype=torch.float32)
        action = torch.tensor([action_features], dtype=torch.float32)
        if state.shape[-1] != int(self.checkpoint["state_feature_dim"]):
            raise ValueError("State feature dimension does not match reward checkpoint")
        if action.shape[-1] != int(self.checkpoint["action_feature_dim"]):
            raise ValueError("Action feature dimension does not match reward checkpoint")
        state = (state - self.checkpoint["state_feature_mean"]) / (self.checkpoint["state_feature_std"] + 1e-6)
        action = (action - self.checkpoint["action_feature_mean"]) / (self.checkpoint["action_feature_std"] + 1e-6)
        with torch.no_grad():
            raw = self.model(state, action, torch.tensor([int(action_id)], dtype=torch.long))
            normalized = (raw - float(self.checkpoint["reward_mean"])) / (float(self.checkpoint["reward_std"]) + 1e-6)
        result = float(normalized.item())
        if not math.isfinite(result):
            raise RewardModelCheckpointError("Reward model produced a non-finite score")
        return result
