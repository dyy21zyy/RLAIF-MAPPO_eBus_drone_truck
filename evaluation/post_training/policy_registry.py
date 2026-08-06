"""Strict registry for only the three isolated post-training policies."""
from __future__ import annotations
from typing import Any
METHODS={
 "mappo_env_post_base":("four_agent_asynchronous_mappo_env_post_base","base_pretraining","none",()),
 "mappo_env_post_continued":("four_agent_asynchronous_mappo_env_post_continued","environment_post_training","none",()),
 "mappo_rlaif_assignment_post":("four_agent_asynchronous_mappo_rlaif_assignment_post","rlaif_post_training","assignment",("assignment",)),}
REQUIRED=("training_seed","reward_scale_artifact_hash","scenario_bank_hash","code_commit","checkpoint_sha256")
def validate_policy(record:dict[str,Any])->dict[str,Any]:
    method=record.get("method_id")
    if method not in METHODS: raise ValueError("unregistered post-training policy")
    alg,stage,scope,agents=METHODS[method]
    for key,want in (("algorithm",alg),("training_stage",stage),("rlaif_scope",scope)):
      if record.get(key)!=want: raise ValueError(f"{method} invalid {key}")
    if tuple(record.get("enabled_reward_agents",()))!=agents: raise ValueError("enabled reward-agent lineage mismatch")
    if record.get("training_seed") not in (1,2,3): raise ValueError("invalid training seed")
    for key in REQUIRED:
      if record.get(key) in (None,""): raise ValueError(f"missing policy lineage: {key}")
    if method.endswith("post_continued") or method.endswith("assignment_post"):
      if not record.get("parent_checkpoint_sha256"): raise ValueError("child policy lacks parent lineage")
    if method.endswith("assignment_post") and not record.get("reward_model_sha256"): raise ValueError("RLAIF policy lacks reward-model lineage")
    return record
