"""Learned-reward checkpoint adapter with validation, fallback, and clipping gates."""

from __future__ import annotations

import importlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Sequence

LOGGER = logging.getLogger(__name__)


class RewardModelCheckpointError(RuntimeError):
    """Raised when learned RLAIF reward was requested but cannot be loaded safely."""


class RewardModelWrapper:
    """Score assignment-level objective features for RLAIF reward shaping.

    RLAIF is intentionally assignment-scoped in this repository. Truck, bus, and
    station rewards must continue to use only environment rewards unless a future
    implementation adds real multi-agent preference labels and integrations.
    """

    REQUIRED_KEYS = {
        "model_state_dict", "state_feature_dim", "action_feature_dim", "num_actions",
        "state_feature_mean", "state_feature_std", "action_feature_mean", "action_feature_std",
        "reward_mean", "reward_std",
    }

    def __init__(
        self,
        checkpoint_path: str | Path | None,
        *,
        enabled: bool,
        validation: dict[str, Any] | None = None,
        fallback_to_env_reward: bool = False,
        fail_on_invalid_reward_model: bool = True,
        reward_clip: float | None = None,
    ) -> None:
        self.requested_enabled = bool(enabled)
        self.enabled = False
        self.active = False
        self.fallen_back = False
        self.fallback_reason = ""
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model = None
        self.checkpoint = None
        self.validation = dict(validation or {})
        self.fallback_to_env_reward = bool(fallback_to_env_reward)
        self.fail_on_invalid_reward_model = bool(fail_on_invalid_reward_model)
        self.reward_clip = None if reward_clip is None else abs(float(reward_clip))
        if not self.requested_enabled:
            LOGGER.info("RLAIF disabled; using pure environment reward.")
            return
        try:
            self._load()
            self.enabled = True
            self._validate_if_requested()
            self.active = True
            LOGGER.info("RLAIF active for assignment-agent decisions only%s.",
                        " with reward clipping" if self.reward_clip is not None else "")
        except Exception as exc:
            if isinstance(exc, RewardModelCheckpointError):
                error = exc
            else:
                error = RewardModelCheckpointError(str(exc))
            if self.fallback_to_env_reward and not self.fail_on_invalid_reward_model:
                self.fallen_back = True
                self.fallback_reason = str(error)
                LOGGER.warning("RLAIF reward model unavailable/invalid; falling back to pure environment reward: %s", error)
                self.model = None; self.checkpoint = None; self.enabled = False; self.active = False
                return
            raise error

    def _load(self) -> None:
        if self.checkpoint_path is None or not self.checkpoint_path.is_file():
            raise RewardModelCheckpointError(
                f"rlaif.enabled=true requires a valid Stage 5 reward model checkpoint: {self.checkpoint_path}"
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
            raise RewardModelCheckpointError(f"Invalid Stage 5 reward model checkpoint {self.checkpoint_path}: {exc}") from exc
        self.torch = torch; self.checkpoint = checkpoint; self.model = model

    def _validate_if_requested(self) -> None:
        if not bool(self.validation.get("enabled", False)):
            return
        path = self.validation.get("preference_data")
        min_samples = int(self.validation.get("min_samples", 1))
        min_acc = float(self.validation.get("min_pairwise_accuracy", 0.0))
        if not path or not Path(path).is_file():
            raise RewardModelCheckpointError(f"RLAIF validation preference_data is missing: {path}")
        total = correct = 0
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                state = row.get("state_features")
                chosen = row.get("chosen_action_features")
                rejected = row.get("rejected_action_features")
                chosen_id = int(row.get("chosen_action_id", 0))
                rejected_id = int(row.get("rejected_action_id", 0))
                if state is None or chosen is None or rejected is None:
                    continue
                total += 1
                correct += int(self.score(state, chosen, chosen_id) > self.score(state, rejected, rejected_id))
        if total < min_samples:
            raise RewardModelCheckpointError(f"RLAIF validation found {total} samples, below min_samples={min_samples}")
        acc = correct / max(total, 1)
        if acc < min_acc:
            raise RewardModelCheckpointError(f"RLAIF validation pairwise accuracy {acc:.3f} below min_pairwise_accuracy={min_acc:.3f}")

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
        if self.reward_clip is not None:
            result = max(-self.reward_clip, min(self.reward_clip, result))
        if not math.isfinite(result):
            raise RewardModelCheckpointError("Reward model produced a non-finite score")
        return result
