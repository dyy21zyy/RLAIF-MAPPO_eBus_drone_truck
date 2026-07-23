
import json, hashlib, pytest
from pathlib import Path
from evaluation.artifact_inventory import build_artifact_inventory, canonical_json_hash
from evaluation.preformal_part2_gates import *

def _write_json(p, payload):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(payload, sort_keys=True), encoding='utf-8'); return p

def _manifest(p, split, ids):
    scenarios=[{'scenario_id':i,'seed_tuple':{'base_seed':n,'dynamic_seed':n+100},'scenario_content_hash':'dyn_'+i,'instance_hash':'inst_'+i,'artifact_hashes':{'road_graph':'shared_static'}} for n,i in enumerate(ids, start={'train':0,'validation':1000,'test':2000}.get(split,0))]
    return _write_json(p, {'schema_version':1,'split':split,'run_classification':'formal','scenarios':scenarios})

def test_reward_stages_are_distinct_and_reconcile():
    t={'agent':'assignment','environment_reward':10,'lambda':2,'raw_learned_reward':3,'normalized_learned_reward':1.5,'clipped_learned_reward':1,'weighted_learned_reward':2,'combined_reward':12}
    assert len({t['raw_learned_reward'],t['normalized_learned_reward'],t['clipped_learned_reward'],t['weighted_learned_reward'],t['combined_reward']})==5
    assert validate_reward_decomposition([t])['validation_status']=='passed'
