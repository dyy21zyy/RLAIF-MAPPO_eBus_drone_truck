import torch, pytest
from tests.real_readiness_helpers import run_pilot, load
from training.mappo_trainer import load_checkpoint

def test_checkpoint_roundtrip(tmp_path):
    out=run_pilot(tmp_path); r=load(out,"checkpoint_roundtrip_report.json")
    assert r["passed"] and r["metadata"]["contains_actor_weights"] and r["metadata"]["contains_critic_weights"]
    actors, critic, meta = load_checkpoint(r["checkpoint_path"]); assert actors and critic and meta

def test_schema_incompatibility_fails(tmp_path):
    out=run_pilot(tmp_path); p=load(out,"checkpoint_roundtrip_report.json")["checkpoint_path"]
    ck=torch.load(p, map_location="cpu", weights_only=False); ck["event_schema_version"]=-1; bad=tmp_path/"bad.pt"; torch.save(ck,bad)
    with pytest.raises(ValueError): load_checkpoint(bad)
