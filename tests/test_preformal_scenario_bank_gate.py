
import json, hashlib, pytest
from pathlib import Path
from evaluation.artifact_inventory import build_artifact_inventory, canonical_json_hash
from evaluation.preformal_part2_gates import *

def _write_json(p, payload):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(payload, sort_keys=True), encoding='utf-8'); return p

def _manifest(p, split, ids):
    scenarios=[{'scenario_id':i,'seed_tuple':{'base_seed':n,'dynamic_seed':n+100},'scenario_content_hash':'dyn_'+i,'instance_hash':'inst_'+i,'artifact_hashes':{'road_graph':'shared_static'}} for n,i in enumerate(ids, start={'train':0,'validation':1000,'test':2000}.get(split,0))]
    return _write_json(p, {'schema_version':1,'split':split,'run_classification':'formal','scenarios':scenarios})

def test_disjointness_passes_and_shared_static_passes(tmp_path):
    r=validate_scenario_bank_splits(_manifest(tmp_path/'train.json','train',['a','b']), _manifest(tmp_path/'val.json','validation',['c']), _manifest(tmp_path/'test.json','test',['d']))
    assert r['validation_status']=='passed' and r['shared_static_network_allowed']

def test_duplicate_dynamic_hash_fails(tmp_path):
    tr=_manifest(tmp_path/'train.json','train',['a']); va=_manifest(tmp_path/'val.json','validation',['b']); te=_manifest(tmp_path/'test.json','test',['c'])
    m=json.loads(va.read_text()); m['scenarios'][0]['scenario_content_hash']='dyn_a'; va.write_text(json.dumps(m))
    with pytest.raises(ScenarioSplitLeakageError): validate_scenario_bank_splits(tr,va,te)
