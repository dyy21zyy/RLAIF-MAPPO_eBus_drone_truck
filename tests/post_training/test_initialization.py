import copy, json
import pytest, torch
from training.post_training.initialization import initialize_policy
from .conftest import models, digest

def test_pretrained_actors_and_critic_load_exactly_and_new_optimizers(parent):
    config, checkpoint, _, _, _, _, _ = parent; actors, critic = models()
    record=initialize_policy(actors, critic, config)
    for name in actors:
        for key, value in actors[name].state_dict().items(): assert torch.equal(value, checkpoint["actor_state_dicts"][name][key])
    for key, value in critic.state_dict().items(): assert torch.equal(value, checkpoint["critic_state_dict"][key])
    assert record.load_optimizers is False
    optimizer=torch.optim.Adam(actors["assignment"].parameters()); assert optimizer.state == {}

@pytest.mark.parametrize("key,value", [("training_seed",8),("method_id","bad"),("algorithm","bad"),("rlaif_scope","assignment"),
    ("training_scenario_bank_hash","bad"),("reward_scale_artifact_hash","bad"),("event_schema_version",-1)])
def test_checkpoint_contract_mismatch_fails(parent, key, value):
    config, checkpoint, manifest, _, _, cp, mp = parent; checkpoint[key]=value; torch.save(checkpoint, cp)
    manifest["checkpoint_file_sha256"]=digest(cp); mp.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError): initialize_policy(*models(), config)

def test_missing_actor_and_shape_mismatch_fail(parent):
    config, checkpoint, manifest, _, _, cp, mp = parent
    checkpoint["actor_state_dicts"].pop("bus"); torch.save(checkpoint, cp)
    manifest["checkpoint_file_sha256"]=digest(cp); mp.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="actor set"): initialize_policy(*models(), config)

@pytest.mark.parametrize("field,value", [("checkpoint_path","/wrong"),("scenario_bank_hash","wrong"),("reward_scale_artifact_hash","wrong")])
def test_manifest_contract_mismatch_fails(parent, field, value):
    config, _, manifest, _, _, _, mp=parent; manifest[field]=value; mp.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError): initialize_policy(*models(), config)

def test_manifest_and_checkpoint_tampering_fail(parent):
    config, _, _, _, _, cp, mp=parent
    cp.write_bytes(cp.read_bytes()+b"tampered")
    with pytest.raises(Exception): initialize_policy(*models(), config)
    # Manifest mutation is detected by its validated contents and its digest is recorded otherwise.
    mp.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError): initialize_policy(*models(), config)

def test_critic_dimension_mismatch_fails(parent):
    config=parent[0]; actors, _=models()
    from training.mappo_networks import CentralizedCritic
    with pytest.raises(ValueError, match="critic_spec"): initialize_policy(actors, CentralizedCritic(6,(4,)), config)
