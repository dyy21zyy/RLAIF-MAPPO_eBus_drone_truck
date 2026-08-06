"""Fail-closed fairness and lineage gates."""
from __future__ import annotations
from typing import Any
FAIR_FIELDS=("total_episodes","rollout_episodes","lr_actor","lr_critic","gamma","gae_lambda","clip_eps","ppo_epochs","batch_size","entropy_coef","value_coef","max_grad_norm","event_time_reference_min","device_policy","scenario_bank","reward_scale_artifact","network_architecture")
INIT_FIELDS=("load_actors","load_critic","load_optimizers")
class ReadinessError(ValueError): pass

def _nested(cfg:dict[str,Any],key:str):
    t=cfg.get("training",{}); return t.get(key,cfg.get(key))
def validate_seed_pair(base:dict[str,Any],continued:dict[str,Any],rlaif:dict[str,Any])->None:
    seeds=[_nested(x,"seed") for x in (base,continued,rlaif)]
    if len(set(seeds))!=1: raise ReadinessError(f"training seed mismatch: {seeds}")
    base_hash=base.get("checkpoint_sha256")
    if not base_hash or continued.get("parent_checkpoint_sha256")!=base_hash or rlaif.get("parent_checkpoint_sha256")!=base_hash: raise ReadinessError("child parent hashes must equal the base checkpoint hash")
    for key in FAIR_FIELDS:
      if _nested(continued,key)!=_nested(rlaif,key): raise ReadinessError(f"post-training fairness mismatch: {key}")
    ci=continued.get("training",{}).get("initialization",{}); ri=rlaif.get("training",{}).get("initialization",{})
    for key in INIT_FIELDS:
      if ci.get(key)!=ri.get(key): raise ReadinessError(f"initialization fairness mismatch: {key}")
    if ci.get("load_optimizers") is not False or ri.get("load_optimizers") is not False: raise ReadinessError("child optimizers must be reset")
    cr=continued.get("rlaif",{})
    if cr.get("enabled") is not False or cr.get("reward_model_path") or cr.get("agents"): raise ReadinessError("continued control must not carry a Reward Model")
    rr=rlaif.get("rlaif",{})
    if rr.get("enabled") is not True or rr.get("scope")!="assignment" or rr.get("validation_status") not in {"pass","passed","validated"}: raise ReadinessError("invalid assignment Reward Model")
    agents=rr.get("agents",{})
    if not agents.get("assignment",{}).get("enabled") or any(agents.get(a,{}).get("enabled") for a in ("truck","bus","station")): raise ReadinessError("only assignment Reward Model may be enabled")
