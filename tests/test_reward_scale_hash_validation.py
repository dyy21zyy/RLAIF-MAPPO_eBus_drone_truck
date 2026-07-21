import json, hashlib, pytest
from pathlib import Path
from envs.reward_scales import load_reward_scale_artifact

def write_art(p, scales):
    payload={'artifact_version':1,'scales':scales}
    payload['artifact_hash']=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    p.write_text(json.dumps(payload)); return payload

def test_scale_hash_missing_component_and_missing_file_fail(tmp_path):
    with pytest.raises(FileNotFoundError): load_reward_scale_artifact(tmp_path/'x.json', expected_hash=None, required_components={'a'})
    p=tmp_path/'s.json'; payload=write_art(p, {'a':2.0})
    assert load_reward_scale_artifact(p, expected_hash=payload['artifact_hash'], required_components={'a'}).scales['a']==2.0
    with pytest.raises(ValueError): load_reward_scale_artifact(p, expected_hash='bad', required_components={'a'})
    with pytest.raises(ValueError): load_reward_scale_artifact(p, expected_hash=None, required_components={'b'})
    bad=tmp_path/'bad.json'; bad.write_text(json.dumps({'artifact_version':1,'scales':{'a':1},'artifact_hash':'placeholder'}))
    with pytest.raises(ValueError): load_reward_scale_artifact(bad, expected_hash=None, required_components={'a'})
