import torch
from training.post_training.checkpoint import build_post_training_checkpoint
from training.post_training.initialization import PolicyInitializationRecord
from .conftest import models

def test_base_checkpoint_has_null_parent_fields():
    actors,critic=models(); opts={k:torch.optim.Adam(v.parameters()) for k,v in actors.items()}; copt=torch.optim.Adam(critic.parameters())
    config={"method_id":"mappo_env_post_base","algorithm":"four_agent_asynchronous_mappo_env_post_base","training_stage":"base_pretraining",
            "training":{"seed":1},"training_scenario_bank_hash":"s","reward_scale_artifact_hash":"r","code_commit":"abc"}
    payload=build_post_training_checkpoint(config=config,actors=actors,critic=critic,actor_optimizers=opts,critic_optimizer=copt,
        initialization=PolicyInitializationRecord("random","base_pretraining"),optimizer_updates=1)
    assert payload["post_training_checkpoint_schema_version"]==1
    assert all(value is None for key,value in payload.items() if key.startswith("parent_"))
