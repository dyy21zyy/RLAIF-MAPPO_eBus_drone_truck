from __future__ import annotations
import json
from pathlib import Path
from envs.reward_components import REWARD_COMPONENTS
from envs.reward_scales import canonical_payload_hash

def write_scale_artifact(path: Path, *, run_classification='formal', bank_hash='bank'):
    comps={c:{'scale':1.0,'status':'observed_positive','positive_count':1,'minimum_override':None} for c in REWARD_COMPONENTS}
    payload={'artifact_type':'reward_reference_scales','artifact_version':1,'run_classification':run_classification,'validation_status':'passed','component_order':list(REWARD_COMPONENTS),'training_scenario_bank_hash':bank_hash,'training_scenario_count':1,'reference_policy_suite':[{'name':'truck_direct_reference','version':1}],'estimator':{'method':'percentile','percentile':95},'components':comps,'scales':{c:1.0 for c in REWARD_COMPONENTS},'source_episode_file_hash':'e','statistics_file_hash':'s','resolved_config_hash':'c','code_commit':'test','creation_timestamp':'2026-07-22T00:00:00Z'}
    payload['artifact_hash']=canonical_payload_hash(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload
