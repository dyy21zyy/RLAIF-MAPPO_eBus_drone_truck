from evaluation.scenario_bank import *
def test_manifest_unique(tmp_path):
    p=tmp_path/'i.json'; p.write_text('{"ok":1}')
    s=freeze_scenario(p,tmp_path/'test'/'s1','s1',{'a':1},{})
    m=write_bank_manifest(tmp_path/'test','test',[s],{})
    assert m['scenario_ids']==['s1'] and m['bank_hash']
def test_hash_verification(tmp_path):
    p=tmp_path/'i.json'; p.write_text('{"ok":1}')
    s=freeze_scenario(p,tmp_path/'test'/'s1','s1',{'a':1},{})
    fs=FrozenScenario('s1','test',s['instance_path'],s['scenario_manifest_path'],s['artifact_hashes'],{},'')
    verify_scenario_hashes(fs)
