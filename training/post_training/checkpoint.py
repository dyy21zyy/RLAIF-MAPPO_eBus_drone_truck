"""Independent checkpoint schema for policy post-training."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import torch

from .initialization import PolicyInitializationRecord

SCHEMA_VERSION = 1


def build_post_training_checkpoint(*, config: dict[str, Any], actors, critic, actor_optimizers, critic_optimizer,
                                   initialization: PolicyInitializationRecord, optimizer_updates: int,
                                   reward_checkpoint_paths: dict[str, str] | None = None,
                                   reward_checkpoint_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    if optimizer_updates <= 0:
        raise ValueError("a completed post-training checkpoint requires optimizer_updates > 0")
    record = initialization
    payload = {
        "post_training_checkpoint_schema_version": SCHEMA_VERSION,
        "method_id": config["method_id"], "algorithm": config["algorithm"],
        "training_stage": config["training_stage"], "initialization_mode": record.initialization_mode,
        "training_seed": int(config["training"]["seed"]),
        "parent_method_id": record.parent_method_id, "parent_algorithm": record.parent_algorithm,
        "parent_training_seed": record.parent_training_seed,
        "parent_checkpoint_path": record.parent_checkpoint_path,
        "parent_checkpoint_sha256": record.parent_checkpoint_sha256,
        "parent_manifest_path": record.parent_manifest_path,
        "parent_manifest_sha256": record.parent_manifest_sha256,
        "initialization_load_actors": record.load_actors, "initialization_load_critic": record.load_critic,
        "initialization_load_optimizers": record.load_optimizers,
        "actor_state_dicts": {key: value.state_dict() for key, value in actors.items()},
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dicts": {key: value.state_dict() for key, value in actor_optimizers.items()},
        "critic_optimizer_state_dict": critic_optimizer.state_dict(), "optimizer_updates": optimizer_updates,
        "training_scenario_bank_hash": config["training_scenario_bank_hash"],
        "reward_scale_artifact_hash": config["reward_scale_artifact_hash"],
        "reward_checkpoint_paths": reward_checkpoint_paths or {}, "reward_checkpoint_hashes": reward_checkpoint_hashes or {},
        "code_commit": config["code_commit"],
    }
    parent_fields = [key for key in payload if key.startswith("parent_")]
    if record.initialization_mode == "random" and any(payload[key] is not None for key in parent_fields):
        raise ValueError("base pretraining parent fields must be null")
    if record.initialization_mode == "pretrained_policy" and any(payload[key] is None for key in parent_fields):
        raise ValueError("post-training child parent fields must be complete")
    return payload


def save_post_training_checkpoint(path: str | Path, **kwargs: Any) -> Path:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_post_training_checkpoint(**kwargs), target)
    return target
