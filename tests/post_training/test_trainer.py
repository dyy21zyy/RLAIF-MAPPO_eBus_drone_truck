import pytest, torch
from training.post_training.trainer import clipped_policy_loss, combine_post_training_reward, prepare_post_training, validate_frozen_assignment_reward_model
from training.mappo_trainer import update_mappo
from .conftest import models, digest

def test_assignment_only_reward_composition():
    assert combine_post_training_reward("assignment", 2, 3, .5)==3.5
    assert combine_post_training_reward("truck", 2, 99, .5)==2

def test_ppo_surrogate_parity_with_legacy_expression():
    new=torch.tensor([-.1,.2]); old=torch.tensor([-.2,.1]); adv=torch.tensor([1.,-2.]); clip=.2
    ratio=torch.exp(new-old); expected=-torch.minimum(ratio*adv, torch.clamp(ratio,1-clip,1+clip)*adv).mean()
    assert torch.equal(clipped_policy_loss(new,old,adv,clip), expected)

def test_prepare_creates_fresh_optimizer_state(parent):
    actors,critic=models(); _, actor_opts, critic_opt=prepare_post_training(actors,critic,parent[0])
    assert all(opt.state == {} for opt in actor_opts.values()); assert critic_opt.state == {}

def test_non_assignment_reward_model_rejected(tmp_path):
    path=tmp_path/"rm.pt"; path.write_bytes(b"model")
    config={"rlaif":{"reward_model_path":str(path),"normalization":{"std":1},"clip":[-1,1]},"observation_schema_version":1,"candidate_schema_version":1}
    metadata={"validation_status":"passed","agent_type":"truck","observation_schema_version":1,"candidate_schema_version":1,"checkpoint_sha256":digest(path)}
    with pytest.raises(ValueError, match="assignment"): validate_frozen_assignment_reward_model(torch.nn.Linear(1,1),metadata,config)

def test_legacy_trainer_is_not_imported_as_post_trainer():
    assert update_mappo.__module__ == "training.mappo_trainer"
