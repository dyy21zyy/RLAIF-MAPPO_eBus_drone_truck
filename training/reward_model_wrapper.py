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


class RewardCheckpointError(RuntimeError):
    """Base exception for strict reward-checkpoint loading."""

class RewardCheckpointCompatibilityError(RewardCheckpointError):
    """Checkpoint exists but is incompatible with requested runtime schemas."""

class RewardCheckpointValidationError(RewardCheckpointError):
    """Checkpoint is not formally validated for RLAIF use."""

def load_strict_agent_reward_checkpoint(path: str | Path, *, agent_type: str, expected_event_types: Sequence[str] | None = None, expected_state_feature_names: Sequence[str] | None = None, expected_candidate_feature_names: Sequence[str] | None = None, expected_preference_file_hash: str | None = None, formal: bool = True):
    import torch
    from training.event_schema import OBSERVATION_SCHEMA_VERSION, CANDIDATE_SCHEMA_VERSION, EVENT_SCHEMA_VERSION, EVENT_NAME_TO_ID, REQUIRED_EVENT_COVERAGE
    from rlaif.multi_agent_reward_model import AgentRewardModel
    path=Path(path)
    if not path.is_file(): raise RewardCheckpointError(f"Reward checkpoint does not exist: {path}")
    if path.suffix.lower()=='.json':
        try:
            data=json.loads(path.read_text())
        except Exception as exc: raise RewardCheckpointError(f"Invalid reward checkpoint JSON: {exc}") from exc
        if data.get('smoke_placeholder') is True: raise RewardCheckpointValidationError('The supplied artifact is a legacy placeholder summary, not a trained reward-model checkpoint.')
        raise RewardCheckpointError('The supplied artifact is JSON, not a PyTorch reward-model checkpoint.')
    try: ck=torch.load(path,map_location='cpu',weights_only=False)
    except Exception as exc: raise RewardCheckpointError(f"Unable to load PyTorch reward checkpoint {path}: {exc}") from exc
    if not isinstance(ck,dict): raise RewardCheckpointError('checkpoint is not a dictionary')
    if ck.get('checkpoint_type')!='agent_reward_model': raise RewardCheckpointCompatibilityError('checkpoint_type is not agent_reward_model')
    if ck.get('checkpoint_schema_version') not in {1}: raise RewardCheckpointCompatibilityError('unsupported reward checkpoint schema version')
    if formal and ck.get('run_classification')!='formal': raise RewardCheckpointValidationError('formal loading requires run_classification = formal')
    if formal and ck.get('validation_status')!='passed': raise RewardCheckpointValidationError('formal loading requires validation_status = passed')
    if ck.get('agent_type')!=agent_type: raise RewardCheckpointCompatibilityError(f"checkpoint agent_type {ck.get('agent_type')} does not match requested {agent_type}")
    req=tuple(expected_event_types or sorted(REQUIRED_EVENT_COVERAGE[agent_type]))
    if set(ck.get('compatible_event_types',[])) != set(req): raise RewardCheckpointCompatibilityError('compatible event types do not match')
    if ck.get('observation_schema_version')!=OBSERVATION_SCHEMA_VERSION or ck.get('candidate_schema_version')!=CANDIDATE_SCHEMA_VERSION or ck.get('event_schema_version')!=EVENT_SCHEMA_VERSION: raise RewardCheckpointCompatibilityError('schema version mismatch')
    if ck.get('event_name_to_id')!=dict(EVENT_NAME_TO_ID): raise RewardCheckpointCompatibilityError('event_name_to_id mismatch')
    if expected_state_feature_names is not None and list(expected_state_feature_names)!=ck.get('state_feature_names'): raise RewardCheckpointCompatibilityError('state feature names/order mismatch')
    if expected_candidate_feature_names is not None and list(expected_candidate_feature_names)!=ck.get('candidate_feature_names'): raise RewardCheckpointCompatibilityError('candidate feature names/order mismatch')
    if int(ck.get('state_feature_dim',-1))!=len(ck.get('state_feature_names',[])) or int(ck.get('candidate_feature_dim',-1))!=len(ck.get('candidate_feature_names',[])): raise RewardCheckpointCompatibilityError('feature dimensions do not match feature names')
    for k in ('state_normalization_mean','state_normalization_std','candidate_normalization_mean','candidate_normalization_std'):
        vals=ck.get(k);
        if not vals or not all(math.isfinite(float(x)) for x in vals): raise RewardCheckpointCompatibilityError(f'{k} missing or nonfinite')
    if not math.isfinite(float(ck.get('reward_output_training_mean',float('nan')))) or float(ck.get('reward_output_training_std',0))<=0: raise RewardCheckpointCompatibilityError('reward output normalization invalid')
    if expected_preference_file_hash is not None and ck.get('preference_file_hash')!=expected_preference_file_hash: raise RewardCheckpointCompatibilityError('preference_file_hash mismatch')
    arch=ck.get('model_architecture',{})
    model=AgentRewardModel(state_dim=int(ck['state_feature_dim']),candidate_dim=int(ck['candidate_feature_dim']),num_event_types=len(ck['event_name_to_id']),event_embedding_dim=int(arch.get('event_embedding_dim',16)),hidden_dims=tuple(arch.get('hidden_dims',[64,64])),dropout=float(arch.get('dropout',0.0)))
    model.load_state_dict(ck.get('model_state_dict',{}), strict=True); model.eval()
    return ck, model


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
        agent_type: str = "assignment",
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
        self.agent_type = str(agent_type)
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
            from rlaif.multi_agent_reward_model import MultiAgentRewardModel

            if self.checkpoint_path.suffix.lower()=='.json' and json.loads(self.checkpoint_path.read_text()).get('smoke_placeholder') is True:
                raise ValueError('The supplied artifact is a legacy placeholder summary, not a trained reward-model checkpoint.')
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, dict):
                raise ValueError("checkpoint is not a dictionary")
            if checkpoint.get("agent_type") is not None:
                if checkpoint.get("agent_type") != self.agent_type:
                    raise ValueError(f"checkpoint agent_type {checkpoint.get('agent_type')} does not match requested {self.agent_type}")
                if checkpoint.get("state_schema_version") not in {"v2", 2}:
                    raise ValueError("unsupported state_schema_version")
                for key in ("model_state_dict","state_feature_dim","candidate_feature_dim","state_feature_mean","state_feature_std","candidate_feature_mean","candidate_feature_std","reward_mean","reward_std","compatible_event_types"):
                    if key not in checkpoint: raise ValueError(f"missing checkpoint key: {key}")
                model_config = checkpoint.get("model_config", checkpoint.get("config", {}).get("reward_model", {}))
                model = MultiAgentRewardModel(int(checkpoint["state_feature_dim"]), int(checkpoint["candidate_feature_dim"]), checkpoint["compatible_event_types"], hidden_dims=model_config.get("hidden_dims", [64,64]), dropout=float(model_config.get("dropout",0.0)))
                model.load_state_dict(checkpoint["model_state_dict"]); model.eval(); self.torch=torch; self.checkpoint=checkpoint; self.model=model; return
            if self.REQUIRED_KEYS - checkpoint.keys():
                missing = sorted(self.REQUIRED_KEYS - set(checkpoint))
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

    def score(self, state_features: Sequence[float], action_features: Sequence[float], action_id: int, event_type: str | None = None) -> float:
        if not self.enabled or self.model is None or self.checkpoint is None:
            raise RuntimeError("RLAIF scoring is disabled; no substitute reward is available")
        torch = self.torch
        state = torch.tensor([state_features], dtype=torch.float32)
        action = torch.tensor([action_features], dtype=torch.float32)
        cand_key = "candidate_feature_dim" if "candidate_feature_dim" in self.checkpoint else "action_feature_dim"
        if event_type is not None and "compatible_event_types" in self.checkpoint and event_type not in self.checkpoint["compatible_event_types"]:
            raise RewardModelCheckpointError(f"event {event_type} is not compatible with checkpoint for {self.agent_type}")
        if state.shape[-1] != int(self.checkpoint["state_feature_dim"]):
            raise ValueError("State feature dimension does not match reward checkpoint")
        if action.shape[-1] != int(self.checkpoint[cand_key]):
            raise ValueError("Action feature dimension does not match reward checkpoint")
        state = (state - self.checkpoint["state_feature_mean"]) / (self.checkpoint["state_feature_std"] + 1e-6)
        action_mean_key = "candidate_feature_mean" if "candidate_feature_mean" in self.checkpoint else "action_feature_mean"
        action_std_key = "candidate_feature_std" if "candidate_feature_std" in self.checkpoint else "action_feature_std"
        action = (action - self.checkpoint[action_mean_key]) / (self.checkpoint[action_std_key] + 1e-6)
        with torch.no_grad():
            if "compatible_event_types" in self.checkpoint:
                eid = self.checkpoint["compatible_event_types"].index(event_type or self.checkpoint["compatible_event_types"][0])
                raw = self.model(state, action, torch.tensor([eid], dtype=torch.long))
            else:
                raw = self.model(state, action, torch.tensor([int(action_id)], dtype=torch.long))
            normalized = (raw - float(self.checkpoint["reward_mean"])) / (float(self.checkpoint["reward_std"]) + 1e-6)
        result = float(normalized.item())
        if self.reward_clip is not None:
            result = max(-self.reward_clip, min(self.reward_clip, result))
        if not math.isfinite(result):
            raise RewardModelCheckpointError("Reward model produced a non-finite score")
        return result
