"""Strict parent-policy validation and loading for post-training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from training.mappo_async import CANDIDATE_SCHEMA_VERSION, EVENT_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION
from training.entity_encoders import ENTITY_ENCODER_SCHEMA_VERSION
from training.mappo_networks import EVENT_EMBEDDING_SCHEMA_VERSION, EVENT_NAME_TO_ID

ACTOR_IDS = frozenset({"assignment", "truck", "bus", "station"})
PARENT_METHOD = "mappo_env_post_base"
PARENT_ALGORITHM = "four_agent_asynchronous_mappo_env_post_base"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PolicyInitializationRecord:
    initialization_mode: str
    training_stage: str
    parent_method_id: str | None = None
    parent_algorithm: str | None = None
    parent_training_seed: int | None = None
    parent_checkpoint_path: str | None = None
    parent_checkpoint_sha256: str | None = None
    parent_manifest_path: str | None = None
    parent_manifest_sha256: str | None = None
    load_actors: bool = False
    load_critic: bool = False
    load_optimizers: bool = False


def _equal(payload: Mapping[str, Any], key: str, expected: Any, source: str) -> None:
    if payload.get(key) != expected:
        raise ValueError(f"parent {source} {key} mismatch: expected {expected!r}, got {payload.get(key)!r}")


def _model_specs(actors: nn.ModuleDict, critic: nn.Module) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    actor_specs = {
        name: {
            "obs_dim": actor.obs_dim,
            "candidate_feature_dim": actor.candidate_feature_dim,
            "hidden_dims": list(actor.hidden_dims),
            "event_embedding_dim": actor.event_embedding_dim,
        }
        for name, actor in actors.items()
    }
    critic_spec = {"global_state_dim": critic.global_state_dim, "hidden_dims": list(critic.hidden_dims)}
    return actor_specs, critic_spec


def initialize_policy(actors: nn.ModuleDict, critic: nn.Module, config: dict[str, Any], *, device: str | torch.device = "cpu") -> PolicyInitializationRecord:
    """Validate provenance, strictly load policy weights, and prepare train mode."""
    init = config["training"]["initialization"]
    mode = init["mode"]
    if mode == "random":
        actors.to(device).train(); critic.to(device).train()
        return PolicyInitializationRecord(mode, config["training_stage"])
    checkpoint_path = Path(init["checkpoint_path"]).resolve()
    manifest_path = Path(init["manifest_path"]).resolve()
    checkpoint_hash = sha256_file(checkpoint_path)
    manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    seed = int(config["training"]["seed"])
    scenario_hash = config["training_scenario_bank_hash"]
    scale_hash = config["reward_scale_artifact_hash"]
    checks = {
        "post_training_checkpoint_schema_version": 1,
        "method_id": PARENT_METHOD, "algorithm": PARENT_ALGORITHM,
        "training_stage": "base_pretraining", "initialization_mode": "random",
        "rlaif_scope": "none", "enabled_reward_agents": [], "training_seed": seed,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "entity_encoder_schema_version": ENTITY_ENCODER_SCHEMA_VERSION,
        "event_embedding_schema_version": EVENT_EMBEDDING_SCHEMA_VERSION,
        "event_name_to_id": EVENT_NAME_TO_ID,
        "training_scenario_bank_hash": scenario_hash,
        "reward_scale_artifact_hash": scale_hash,
    }
    for key, expected in checks.items(): _equal(checkpoint, key, expected, "checkpoint")
    if not checkpoint.get("code_commit") or int(checkpoint.get("optimizer_updates", 0)) <= 0:
        raise ValueError("parent checkpoint lacks code provenance or completed optimizer updates")
    if set(checkpoint.get("actor_state_dicts", {})) != ACTOR_IDS:
        raise ValueError("parent checkpoint actor set must be exactly assignment, truck, bus, station")
    actor_specs, critic_spec = _model_specs(actors, critic)
    _equal(checkpoint, "actor_specs", actor_specs, "checkpoint")
    _equal(checkpoint, "critic_spec", critic_spec, "checkpoint")
    manifest_checks = {
        "status": "complete", "method_id": PARENT_METHOD, "training_seed": seed,
        "checkpoint_path": str(checkpoint_path), "checkpoint_file_sha256": checkpoint_hash,
        "scenario_bank_hash": scenario_hash, "reward_scale_artifact_hash": scale_hash,
        "reward_model_path": None, "reward_model_hash": None,
    }
    for key, expected in manifest_checks.items(): _equal(manifest, key, expected, "manifest")
    if int(manifest.get("optimizer_updates", 0)) <= 0:
        raise ValueError("parent manifest has no optimizer updates")
    # Individual actor dictionaries make the exact four-agent contract explicit.
    for agent_id in sorted(ACTOR_IDS):
        actors[agent_id].load_state_dict(checkpoint["actor_state_dicts"][agent_id], strict=True)
    critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
    actors.to(device).train(); critic.to(device).train()
    return PolicyInitializationRecord(mode, config["training_stage"], PARENT_METHOD, PARENT_ALGORITHM, seed,
        str(checkpoint_path), checkpoint_hash, str(manifest_path), manifest_hash, True, True, False)
