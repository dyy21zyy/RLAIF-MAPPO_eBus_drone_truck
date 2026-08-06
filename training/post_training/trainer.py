"""Isolated MAPPO post-training primitives; legacy trainer remains untouched."""

from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import torch

from .initialization import initialize_policy, sha256_file


def clipped_policy_loss(new_log_prob, old_log_prob, advantages, clip_ratio: float):
    """PPO surrogate copied from training/mappo_trainer.py (baseline 70b13eb)."""
    ratio = torch.exp(new_log_prob - old_log_prob)
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    return -torch.minimum(ratio * advantages, clipped * advantages).mean()


def combine_post_training_reward(agent_id: str, environment_reward: float, learned_reward: float, weight: float) -> float:
    if not all(math.isfinite(value) for value in (environment_reward, learned_reward, weight)):
        raise ValueError("post-training rewards and weight must be finite")
    return float(environment_reward + (weight * learned_reward if agent_id == "assignment" else 0.0))


def validate_frozen_assignment_reward_model(model, metadata: dict[str, Any], config: dict[str, Any]) -> None:
    path = Path(config["rlaif"]["reward_model_path"]).resolve()
    if metadata.get("validation_status") not in {"passed", "complete", "valid"}:
        raise ValueError("Reward Model validation status is not acceptable")
    if metadata.get("agent_type") != "assignment":
        raise ValueError("post-training accepts only an assignment Reward Model")
    if metadata.get("observation_schema_version") != config["observation_schema_version"] or metadata.get("candidate_schema_version") != config["candidate_schema_version"]:
        raise ValueError("Reward Model feature schema mismatch")
    if metadata.get("checkpoint_sha256") != sha256_file(path):
        raise ValueError("Reward Model checkpoint hash mismatch")
    normalization = config["rlaif"].get("normalization", {})
    if float(normalization.get("std", 0.0)) <= 0 or not math.isfinite(float(normalization.get("std", 0.0))):
        raise ValueError("Reward Model normalization std must be finite and positive")
    clip = config["rlaif"].get("clip")
    if not isinstance(clip, (list, tuple)) or len(clip) != 2 or not float(clip[0]) < float(clip[1]):
        raise ValueError("Reward Model clipping bounds are invalid")
    model.requires_grad_(False); model.eval()
    with torch.inference_mode():
        probe = config["rlaif"].get("validation_probe")
        if probe is not None and not torch.isfinite(model(torch.as_tensor(probe, dtype=torch.float32))).all():
            raise ValueError("Reward Model produced a non-finite validation reward")


def prepare_post_training(actors, critic, config: dict[str, Any], *, device="cpu"):
    """Load models first, then deliberately create entirely fresh optimizers."""
    record = initialize_policy(actors, critic, config, device=device)
    lr = float(config["training"].get("learning_rate", 3e-4))
    actor_optimizers = {name: torch.optim.Adam(actor.parameters(), lr=lr) for name, actor in actors.items()}
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(config["training"].get("critic_learning_rate", lr)))
    return record, actor_optimizers, critic_optimizer
