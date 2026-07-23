
import json, hashlib, pytest
from pathlib import Path
from evaluation.artifact_inventory import build_artifact_inventory, canonical_json_hash
from evaluation.preformal_part2_gates import *

def _write_json(p, payload):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(payload, sort_keys=True), encoding='utf-8'); return p

def _manifest(p, split, ids):
    scenarios=[{'scenario_id':i,'seed_tuple':{'base_seed':n,'dynamic_seed':n+100},'scenario_content_hash':'dyn_'+i,'instance_hash':'inst_'+i,'artifact_hashes':{'road_graph':'shared_static'}} for n,i in enumerate(ids, start={'train':0,'validation':1000,'test':2000}.get(split,0))]
    return _write_json(p, {'schema_version':1,'split':split,'run_classification':'formal','scenarios':scenarios})

def test_inventory_metadata_not_directory_and_tamper(tmp_path):
    p=_write_json(tmp_path/'results/formal/diag.json', {'schema_version':1,'run_classification':'diagnostic','validation_status':'passed'})
    items=build_artifact_inventory({'artifacts':{'train_scenario_bank':{'path':str(p)}}}, strict=True)
    assert items[0].run_classification=='diagnostic' and items[0].validation_status=='failed'
    f=_write_json(tmp_path/'elsewhere/formal.json', {'schema_version':1,'run_classification':'formal','validation_status':'passed'})
    assert build_artifact_inventory({'artifacts':{'train_scenario_bank':{'path':str(f)}}}, strict=True)[0].validation_status=='passed'
    old=items[0].file_hash; p.write_text(p.read_text()+' ')
    assert old != build_artifact_inventory({'artifacts':{'train_scenario_bank':{'path':str(p)}}}, strict=False)[0].file_hash

def test_missing_placeholder_and_invalid_formal(tmp_path):
    items=build_artifact_inventory({'artifacts':{'train_scenario_bank':{'path':str(tmp_path/'missing.json')},'reward_scale_artifact':{'path':'REPLACE_WITH_REAL_PATH'},'assignment_reward_checkpoint':{'path':str(_write_json(tmp_path/'bad.json', {'run_classification':'formal','validation_status':'failed'}))}}}, strict=True)
    by={i.artifact_id:i for i in items}
    assert by['train_scenario_bank'].run_classification=='missing'
    assert by['reward_scale_artifact'].run_classification=='placeholder'
    assert by['assignment_reward_checkpoint'].validation_status=='failed'
