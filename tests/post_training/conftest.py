import hashlib, json
from pathlib import Path
import pytest, torch
from training.entity_encoders import ENTITY_ENCODER_SCHEMA_VERSION
from training.mappo_async import CANDIDATE_SCHEMA_VERSION, EVENT_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION
from training.mappo_networks import EVENT_EMBEDDING_SCHEMA_VERSION, EVENT_NAME_TO_ID, CandidateScoringActor, CentralizedCritic

AGENTS = ("assignment", "truck", "bus", "station")

def models():
    actors = torch.nn.ModuleDict({name: CandidateScoringActor(3, 2, (4,), event_embedding_dim=2) for name in AGENTS})
    return actors, CentralizedCritic(5, (4,))

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@pytest.fixture
def parent(tmp_path):
    actors, critic = models(); checkpoint_path = (tmp_path / "base.pt").resolve()
    actor_specs = {name: {"obs_dim": 3, "candidate_feature_dim": 2, "hidden_dims": [4], "event_embedding_dim": 2} for name in AGENTS}
    checkpoint = {
        "post_training_checkpoint_schema_version": 1, "method_id": "mappo_env_post_base",
        "algorithm": "four_agent_asynchronous_mappo_env_post_base", "training_stage": "base_pretraining",
        "initialization_mode": "random", "rlaif_scope": "none", "enabled_reward_agents": [], "training_seed": 7,
        "optimizer_updates": 2, "code_commit": "abc", "event_schema_version": EVENT_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION, "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "entity_encoder_schema_version": ENTITY_ENCODER_SCHEMA_VERSION,
        "event_embedding_schema_version": EVENT_EMBEDDING_SCHEMA_VERSION, "event_name_to_id": EVENT_NAME_TO_ID,
        "training_scenario_bank_hash": "scenario", "reward_scale_artifact_hash": "scale",
        "actor_specs": actor_specs, "critic_spec": {"global_state_dim": 5, "hidden_dims": [4]},
        "actor_state_dicts": {name: actor.state_dict() for name, actor in actors.items()},
        "critic_state_dict": critic.state_dict(), "actor_optimizer_state_dicts": {"poison": 1},
    }
    torch.save(checkpoint, checkpoint_path)
    manifest_path = (tmp_path / "training_run_manifest.json").resolve()
    manifest = {"status": "complete", "method_id": "mappo_env_post_base", "training_seed": 7,
        "checkpoint_path": str(checkpoint_path), "checkpoint_file_sha256": digest(checkpoint_path), "optimizer_updates": 2,
        "scenario_bank_hash": "scenario", "reward_scale_artifact_hash": "scale", "reward_model_path": None, "reward_model_hash": None}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = {"method_id": "mappo_env_post_continued", "algorithm": "four_agent_asynchronous_mappo_env_post_continued",
        "training_stage": "environment_post_training", "training_scenario_bank_hash": "scenario", "reward_scale_artifact_hash": "scale",
        "code_commit": "child", "training": {"seed": 7, "initialization": {"mode": "pretrained_policy",
        "checkpoint_path": str(checkpoint_path), "manifest_path": str(manifest_path), "load_actors": True,
        "load_critic": True, "load_optimizers": False, "strict": True}}}
    return config, checkpoint, manifest, actors, critic, checkpoint_path, manifest_path
