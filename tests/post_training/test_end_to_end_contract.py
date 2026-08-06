"""Small artifact-level contract: no environment episode or formal training is run."""
import copy
import torch
import pytest
from training.post_training.trainer import combine_post_training_reward, prepare_post_training, validate_frozen_assignment_reward_model
from training.post_training.checkpoint import build_post_training_checkpoint
from training.post_training.initialization import sha256_file
from .conftest import models, digest


def test_children_share_parent_weights_reset_optimizers_and_record_lineage(parent):
    config = parent[0]
    actors_a, critic_a = models(); record_a, opts_a, critic_opt_a = prepare_post_training(actors_a,critic_a,copy.deepcopy(config))
    actors_b, critic_b = models(); record_b, opts_b, critic_opt_b = prepare_post_training(actors_b,critic_b,copy.deepcopy(config))
    assert record_a.parent_checkpoint_sha256 == record_b.parent_checkpoint_sha256
    for agent in actors_a:
        for a,b in zip(actors_a[agent].parameters(),actors_b[agent].parameters()): assert torch.equal(a,b)
    for a,b in zip(critic_a.parameters(),critic_b.parameters()): assert torch.equal(a,b)
    assert all(not opt.state for opt in [*opts_a.values(),*opts_b.values(),critic_opt_a,critic_opt_b])
    child_config={**config,"method_id":"mappo_env_post_continued","algorithm":"child","code_commit":"test"}
    payload=build_post_training_checkpoint(config=child_config,actors=actors_a,critic=critic_a,
      actor_optimizers=opts_a,critic_optimizer=critic_opt_a,initialization=record_a,optimizer_updates=1)
    assert payload["parent_training_seed"] == config["training"]["seed"]
    assert payload["parent_checkpoint_sha256"] == record_a.parent_checkpoint_sha256
    assert payload["initialization_load_optimizers"] is False


def test_assignment_rm_is_frozen_scored_and_hash_tampering_fails_closed(tmp_path):
    path=tmp_path/"assignment_rm.pt"; path.write_bytes(b"validated")
    model=torch.nn.Linear(1,1)
    config={"rlaif":{"reward_model_path":str(path),"normalization":{"std":1},"clip":[-2,2],"validation_probe":[1.]},
      "observation_schema_version":1,"candidate_schema_version":1}
    metadata={"validation_status":"passed","agent_type":"assignment","observation_schema_version":1,
      "candidate_schema_version":1,"checkpoint_sha256":digest(path)}
    validate_frozen_assignment_reward_model(model,metadata,config)
    assert not any(p.requires_grad for p in model.parameters()) and not model.training
    score=float(model(torch.tensor([1.])).item())
    assert combine_post_training_reward("assignment",2.,score,.1) == pytest.approx(2+.1*score)
    assert combine_post_training_reward("truck",2.,score,.1) == 2.
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError,match="hash mismatch"):
      validate_frozen_assignment_reward_model(model,metadata,config)
