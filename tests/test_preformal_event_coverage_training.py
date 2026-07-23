
import json, hashlib, pytest
from pathlib import Path
from evaluation.artifact_inventory import build_artifact_inventory, canonical_json_hash
from evaluation.preformal_part2_gates import *

def _write_json(p, payload):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(payload, sort_keys=True), encoding='utf-8'); return p

def _manifest(p, split, ids):
    scenarios=[{'scenario_id':i,'seed_tuple':{'base_seed':n,'dynamic_seed':n+100},'scenario_content_hash':'dyn_'+i,'instance_hash':'inst_'+i,'artifact_hashes':{'road_graph':'shared_static'}} for n,i in enumerate(ids, start={'train':0,'validation':1000,'test':2000}.get(split,0))]
    return _write_json(p, {'schema_version':1,'split':split,'run_classification':'formal','scenarios':scenarios})

def test_event_coverage_required_events():
    report={'event_count_by_event_type':{e:1 for e in REQUIRED_EVENTS},'decision_count_by_agent':{'assignment':1},'scenario_ids_containing_event':{},'training_transitions_by_agent':{}}
    assert validate_event_coverage(report)['validation_status']=='passed'
    report['event_count_by_event_type']['PARCEL_RELEASE']=0
    with pytest.raises(PreformalGateError): validate_event_coverage(report)
