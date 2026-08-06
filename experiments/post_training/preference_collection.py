"""Assignment-only preference candidate contracts collected from base policies."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable
from .artifacts import sha256_file
REQUIRED_FIELDS={"scenario_id","scenario_content_hash","source_policy_method_id","source_policy_seed","source_policy_checkpoint_path","source_policy_checkpoint_sha256","state_features","chosen_candidate_placeholder","candidate_a","candidate_b","candidate_feature_names","action_mask","event_type","observation_schema_version","candidate_schema_version","collection_code_commit"}
VALID_EVENTS={"TD","TBD","TLD"}

def make_record(*, scenario_id:str,scenario_content_hash:str,seed:int,checkpoint:Path,state_features:list[float],candidate_a:int,candidate_b:int,candidate_feature_names:list[str],action_mask:list[bool],event_type:str,commit:str,split:str)->dict[str,Any]:
    if seed not in (1,2,3): raise ValueError("source seed must be 1, 2, or 3")
    if event_type not in VALID_EVENTS: raise ValueError("invalid assignment action semantics")
    if candidate_a==candidate_b: raise ValueError("candidate actions must differ")
    if min(candidate_a,candidate_b)<0 or max(candidate_a,candidate_b)>=len(action_mask) or not action_mask[candidate_a] or not action_mask[candidate_b]: raise ValueError("candidates must be feasible under action mask")
    return {"agent_type":"assignment","scenario_id":scenario_id,"scenario_content_hash":scenario_content_hash,"dataset_split":split,"source_policy_method_id":"mappo_env_post_base","source_policy_seed":seed,"source_policy_checkpoint_path":str(checkpoint),"source_policy_checkpoint_sha256":sha256_file(checkpoint),"state_features":state_features,"chosen_candidate_placeholder":None,"candidate_a":candidate_a,"candidate_b":candidate_b,"candidate_feature_names":candidate_feature_names,"action_mask":action_mask,"event_type":event_type,"observation_schema_version":4,"candidate_schema_version":4,"collection_code_commit":commit}

def validate_records(rows:Iterable[dict[str,Any]])->list[dict[str,Any]]:
    rows=list(rows); split_by_scenario={}
    for row in rows:
      missing=REQUIRED_FIELDS-row.keys()
      if missing: raise ValueError(f"preference fields missing: {sorted(missing)}")
      if row.get("agent_type")!="assignment" or row.get("source_policy_method_id")!="mappo_env_post_base": raise ValueError("preferences must be assignment-only and sourced from base policy")
      if row["candidate_a"]==row["candidate_b"] or not row["action_mask"][row["candidate_a"]] or not row["action_mask"][row["candidate_b"]]: raise ValueError("invalid candidate pair")
      old=split_by_scenario.setdefault(row["scenario_id"],row["dataset_split"])
      if old!=row["dataset_split"]: raise ValueError("scenario leakage across preference splits")
    return rows
