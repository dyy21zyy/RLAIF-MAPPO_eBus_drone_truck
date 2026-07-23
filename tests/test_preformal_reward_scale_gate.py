
import json, hashlib, pytest
from pathlib import Path
from evaluation.artifact_inventory import build_artifact_inventory, canonical_json_hash
from evaluation.preformal_part2_gates import *

def _write_json(p, payload):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(payload, sort_keys=True), encoding='utf-8'); return p

def _manifest(p, split, ids):
    scenarios=[{'scenario_id':i,'seed_tuple':{'base_seed':n,'dynamic_seed':n+100},'scenario_content_hash':'dyn_'+i,'instance_hash':'inst_'+i,'artifact_hashes':{'road_graph':'shared_static'}} for n,i in enumerate(ids, start={'train':0,'validation':1000,'test':2000}.get(split,0))]
    return _write_json(p, {'schema_version':1,'split':split,'run_classification':'formal','scenarios':scenarios})

def _scale(p, cls='formal', bank='bank', comps=None):
    comps=comps or {c:{'status':'exercised'} for c in REWARD_COMPONENTS}; payload={'artifact_type':'reward_reference_scales','artifact_version':1,'run_classification':cls,'validation_status':'passed','training_scenario_bank_hash':bank,'component_order':list(REWARD_COMPONENTS),'scales':{c:1.0 for c in REWARD_COMPONENTS},'components':comps}; payload['artifact_hash']=canonical_json_hash(payload); return _write_json(p,payload), payload
def test_exact_lineage_passes_and_other_bank_fails(tmp_path):
    p,pay=_scale(tmp_path/'scale.json')
    assert validate_reward_scale_gate(p, expected_hash=pay['artifact_hash'], train_bank_hash='bank')['validation_status']=='passed'
    with pytest.raises(ValueError): validate_reward_scale_gate(p, expected_hash=pay['artifact_hash'], train_bank_hash='other')
def test_diagnostic_missing_unexercised_and_override(tmp_path):
    p,pay=_scale(tmp_path/'diag.json','diagnostic')
    with pytest.raises(ValueError): validate_reward_scale_gate(p, expected_hash=pay['artifact_hash'], train_bank_hash='bank')
    comps={c:{'status':'exercised'} for c in REWARD_COMPONENTS}; comps[REWARD_COMPONENTS[0]]={'status':'unexercised'}; p,pay=_scale(tmp_path/'un.json','formal','bank',comps)
    with pytest.raises(ValueError): validate_reward_scale_gate(p, expected_hash=pay['artifact_hash'], train_bank_hash='bank')
    comps[REWARD_COMPONENTS[0]]={'status':'instrumented_zero'}; p,pay=_scale(tmp_path/'zero.json','formal','bank',comps); d=json.loads(p.read_text()); d['minimum_scale_overrides']={REWARD_COMPONENTS[0]:{'scale':1,'reason':'documented'}}; d['artifact_hash']=canonical_json_hash(d); p.write_text(json.dumps(d))
    assert validate_reward_scale_gate(p, expected_hash=d['artifact_hash'], train_bank_hash='bank')['validation_status']=='passed'
