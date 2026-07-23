
import json, hashlib, pytest
from pathlib import Path
from evaluation.artifact_inventory import build_artifact_inventory, canonical_json_hash
from evaluation.preformal_part2_gates import *

def _write_json(p, payload):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(payload, sort_keys=True), encoding='utf-8'); return p

def _manifest(p, split, ids):
    scenarios=[{'scenario_id':i,'seed_tuple':{'base_seed':n,'dynamic_seed':n+100},'scenario_content_hash':'dyn_'+i,'instance_hash':'inst_'+i,'artifact_hashes':{'road_graph':'shared_static'}} for n,i in enumerate(ids, start={'train':0,'validation':1000,'test':2000}.get(split,0))]
    return _write_json(p, {'schema_version':1,'split':split,'run_classification':'formal','scenarios':scenarios})

def test_environment_mappo_training_evidence():
    assert validate_training_report({'real_environment_episodes_executed':True,'mappo_updates':1,'scenario_ids_used':['a','b'],'training_metrics':{'actor_loss':[1.0],'critic_loss':2.0,'entropy':0.1,'kl':0.0,'grad_norm':1.0},'checkpoint_saved':True,'checkpoint_reloaded':True}, algorithm='mappo_env')['validation_status']=='passed'
    with pytest.raises(PreformalGateError): validate_training_report({'real_environment_episodes_executed':True,'mappo_updates':0,'checkpoint_saved':True,'checkpoint_reloaded':True}, algorithm='mappo_env')
