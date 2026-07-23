
import json, hashlib, pytest
from pathlib import Path
from evaluation.artifact_inventory import build_artifact_inventory, canonical_json_hash
from evaluation.preformal_part2_gates import *

def _write_json(p, payload):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(payload, sort_keys=True), encoding='utf-8'); return p

def _manifest(p, split, ids):
    scenarios=[{'scenario_id':i,'seed_tuple':{'base_seed':n,'dynamic_seed':n+100},'scenario_content_hash':'dyn_'+i,'instance_hash':'inst_'+i,'artifact_hashes':{'road_graph':'shared_static'}} for n,i in enumerate(ids, start={'train':0,'validation':1000,'test':2000}.get(split,0))]
    return _write_json(p, {'schema_version':1,'split':split,'run_classification':'formal','scenarios':scenarios})

def test_assignment_ppo_evidence():
    r=validate_training_report({'real_assignment_events_used':True,'ppo_updates':1,'scenario_ids_used':['a','b'],'training_metrics':{'policy_loss':1,'value_loss':1},'checkpoint_saved':True,'checkpoint_reloaded':True,'action_masks_applied':True,'fixed_baselines_for_non_assignment':True}, algorithm='assignment_ppo')
    assert r['action_masks_applied'] and r['fixed_baselines_for_non_assignment']
