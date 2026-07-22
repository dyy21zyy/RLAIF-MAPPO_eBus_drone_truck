"""Canonical Phase 5B reward-model adapter for runtime RLAIF scoring."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Any
import hashlib
import math

import torch

from training.event_schema import EVENT_NAME_TO_ID, validate_agent_event
from training.reward_model_wrapper import (
    RewardCheckpointCompatibilityError,
    RewardCheckpointError,
    load_strict_agent_reward_checkpoint,
)

EPSILON = 1e-6


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a checkpoint file's bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class RuntimeRewardScore:
    raw_score: float
    normalized_score: float


class RuntimeAgentRewardModel:
    """Load and score a canonical ``agent_reward_model`` checkpoint."""

    def __init__(self, *, checkpoint_path: Path, checkpoint: dict[str, Any], model: torch.nn.Module, checkpoint_hash: str) -> None:
        self.checkpoint_path = checkpoint_path
        self.checkpoint = checkpoint
        self.model = model
        self.checkpoint_hash = checkpoint_hash
        self.agent_type = str(checkpoint["agent_type"])
        self.compatible_event_types = tuple(checkpoint["compatible_event_types"])
        self.state_feature_names = tuple(checkpoint["state_feature_names"])
        self.candidate_feature_names = tuple(checkpoint["candidate_feature_names"])

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        expected_agent_type: str,
        expected_event_types: Sequence[str] | None = None,
        expected_state_feature_names: Sequence[str] | None = None,
        expected_candidate_feature_names: Sequence[str] | None = None,
        expected_checkpoint_hash: str | None = None,
        formal_mode: bool,
    ) -> "RuntimeAgentRewardModel":
        path = Path(checkpoint_path)
        digest = sha256_file(path)
        if expected_checkpoint_hash and expected_checkpoint_hash != digest:
            raise RewardCheckpointCompatibilityError(
                f"reward checkpoint hash mismatch for {path}: expected {expected_checkpoint_hash}, actual {digest}"
            )
        ck, model = load_strict_agent_reward_checkpoint(
            path,
            agent_type=expected_agent_type,
            expected_event_types=expected_event_types,
            expected_state_feature_names=expected_state_feature_names,
            expected_candidate_feature_names=expected_candidate_feature_names,
            formal=formal_mode,
        )
        return cls(checkpoint_path=path, checkpoint=ck, model=model, checkpoint_hash=digest)

    def _vector(self, name: str, values: Sequence[float], expected_dim: int) -> torch.Tensor:
        if len(values) != expected_dim:
            raise RewardCheckpointCompatibilityError(f"{name} dimension mismatch: expected {expected_dim}, actual {len(values)}")
        floats = [float(v) for v in values]
        if not all(math.isfinite(v) for v in floats):
            raise RewardCheckpointCompatibilityError(f"{name} contains nonfinite values")
        return torch.tensor([floats], dtype=torch.float32)

    def _norm(self, tensor: torch.Tensor, mean_key: str, std_key: str) -> torch.Tensor:
        mean = torch.tensor([self.checkpoint[mean_key]], dtype=torch.float32)
        std_values = [float(x) for x in self.checkpoint[std_key]]
        if not all(math.isfinite(x) and x > 0 for x in std_values):
            raise RewardCheckpointCompatibilityError(f"{std_key} must be finite and positive")
        std = torch.tensor([std_values], dtype=torch.float32)
        return (tensor - mean) / (std + EPSILON)

    def score(self, *, state_features: Sequence[float], candidate_features: Sequence[float], event_type: str) -> RuntimeRewardScore:
        event_type = validate_agent_event(self.agent_type, event_type)
        if event_type not in self.compatible_event_types:
            raise RewardCheckpointCompatibilityError(f"event {event_type} is not compatible with checkpoint for {self.agent_type}")
        if self.checkpoint.get("event_name_to_id") != dict(EVENT_NAME_TO_ID):
            raise RewardCheckpointCompatibilityError("checkpoint event_name_to_id does not match runtime event schema")
        state = self._vector("state_features", state_features, int(self.checkpoint["state_feature_dim"]))
        cand = self._vector("candidate_features", candidate_features, int(self.checkpoint["candidate_feature_dim"]))
        state = self._norm(state, "state_normalization_mean", "state_normalization_std")
        cand = self._norm(cand, "candidate_normalization_mean", "candidate_normalization_std")
        event_ids = torch.tensor([EVENT_NAME_TO_ID[event_type]], dtype=torch.long)
        with torch.no_grad():
            raw_tensor = self.model(state, event_ids, cand)
        raw = float(raw_tensor.reshape(-1)[0].item())
        mean = float(self.checkpoint["reward_output_training_mean"])
        std = float(self.checkpoint["reward_output_training_std"])
        if not math.isfinite(raw):
            raise RewardCheckpointError("Reward model produced non-finite raw score")
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise RewardCheckpointCompatibilityError("reward output normalization invalid")
        normalized = (raw - mean) / (std + EPSILON)
        if not math.isfinite(normalized):
            raise RewardCheckpointError("Reward model produced non-finite normalized score")
        return RuntimeRewardScore(raw_score=raw, normalized_score=normalized)
