"""Independent 900-row common-scenario benchmark planner."""
from __future__ import annotations
from typing import Any, Iterable
from . import METHODS,SEEDS
EXPECTED_ROWS=900

def build_rows(policies:dict[tuple[str,int],dict[str,Any]],scenarios:Iterable[dict[str,Any]])->list[dict[str,Any]]:
    scenarios=list(scenarios)
    if len(scenarios)!=100 or len({s["scenario_id"] for s in scenarios})!=100: raise ValueError("benchmark requires exactly 100 unique common scenarios")
    rows=[]
    for method in METHODS:
      for seed in SEEDS:
       p=policies[(method,seed)]
       for scenario in scenarios:
        rows.append({"method_id":method,"algorithm":p["algorithm"],"training_stage":p["training_stage"],"training_seed":seed,"scenario_id":scenario["scenario_id"],"scenario_content_hash":scenario["scenario_content_hash"],"policy_checkpoint_path":p["checkpoint_path"],"policy_checkpoint_sha256":p["checkpoint_sha256"],"parent_checkpoint_path":p.get("parent_checkpoint_path"),"parent_checkpoint_sha256":p.get("parent_checkpoint_sha256"),"reward_checkpoint_paths":p.get("reward_checkpoint_paths",[]),"reward_checkpoint_hashes":p.get("reward_checkpoint_hashes",[]),"reward_scale_artifact_path":p["reward_scale_artifact_path"],"reward_scale_artifact_hash":p["reward_scale_artifact_hash"],"code_commit":p["code_commit"],"formal_metrics":{},"status":"planned"})
    if len(rows)!=EXPECTED_ROWS: raise AssertionError("benchmark Cartesian product is incomplete")
    return rows
