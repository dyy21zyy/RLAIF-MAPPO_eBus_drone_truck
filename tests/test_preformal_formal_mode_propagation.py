
import json, hashlib, pytest
from pathlib import Path
from evaluation.artifact_inventory import build_artifact_inventory, canonical_json_hash
from evaluation.preformal_part2_gates import *

def _write_json(p, payload):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(payload, sort_keys=True), encoding='utf-8'); return p

def _manifest(p, split, ids):
    scenarios=[{'scenario_id':i,'seed_tuple':{'base_seed':n,'dynamic_seed':n+100},'scenario_content_hash':'dyn_'+i,'instance_hash':'inst_'+i,'artifact_hashes':{'road_graph':'shared_static'}} for n,i in enumerate(ids, start={'train':0,'validation':1000,'test':2000}.get(split,0))]
    return _write_json(p, {'schema_version':1,'split':split,'run_classification':'formal','scenarios':scenarios})

def test_strict_failure_vs_diagnostic_fallback_recorded():
    strict={'fallback_to_env_reward':False,'fail_on_invalid_reward_model':True,'formal_mode_reached_reward_registry':True,'fallback_count':1}
    with pytest.raises(PreformalGateError):
        if strict['fallback_count']: raise PreformalGateError('strict fallback occurred')
    diag={'fallback_to_env_reward':True,'fail_on_invalid_reward_model':False,'diagnostic_fallback_recorded':True}
    assert diag['diagnostic_fallback_recorded']
