
import json, hashlib, pytest
from pathlib import Path
from evaluation.artifact_inventory import build_artifact_inventory, canonical_json_hash
from evaluation.preformal_part2_gates import *

def _write_json(p, payload):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(payload, sort_keys=True), encoding='utf-8'); return p

def _manifest(p, split, ids):
    scenarios=[{'scenario_id':i,'seed_tuple':{'base_seed':n,'dynamic_seed':n+100},'scenario_content_hash':'dyn_'+i,'instance_hash':'inst_'+i,'artifact_hashes':{'road_graph':'shared_static'}} for n,i in enumerate(ids, start={'train':0,'validation':1000,'test':2000}.get(split,0))]
    return _write_json(p, {'schema_version':1,'split':split,'run_classification':'formal','scenarios':scenarios})

def test_training_cannot_open_test_manifest(tmp_path):
    test=_manifest(tmp_path/'test.json','test',['z'])
    cfg={'train_manifest':str(tmp_path/'train.json'),'validation_manifest':str(tmp_path/'val.json')}
    r=run_training_with_test_access_guard(lambda c:{'updates':1}, cfg, test)
    assert not r['test_manifest_opened']
    with pytest.raises(PreformalGateError): guard_training_config_excludes_test_bank({'test_manifest':str(test)}, test)
