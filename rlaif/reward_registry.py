from __future__ import annotations
import logging, math
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from training.event_schema import AGENT_TYPES, validate_agent_event
from training.reward_contribution import RewardContribution, build_reward_contribution
from training.reward_model_wrapper import RewardModelCheckpointError, RewardModelWrapper

LOGGER=logging.getLogger(__name__)

class RewardRegistry:
    def __init__(self, config:dict[str,Any]):
        r=config.get("rlaif",config)
        self.enabled=bool(r.get("enabled",False))
        self.fallback=bool(r.get("fallback_to_env_reward",False))
        self.fail=bool(r.get("fail_on_invalid_reward_model",True))
        if self.enabled and self.fail and self.fallback:
            raise ValueError("formal RLAIF mode cannot enable silent fallback_to_env_reward")
        self.agents:dict[str,dict[str,Any]]={}; self.wrappers={}
        for agent,cfg in r.get("agents",{}).items():
            if agent not in AGENT_TYPES: raise ValueError(f"Unknown RLAIF agent setting: {agent}")
            lam=float(cfg.get("lambda", r.get("lambda", 0.0)))
            clip=float(cfg.get("reward_clip", r.get("reward_clip", 2.0)))
            if not math.isfinite(lam) or lam < 0: raise ValueError("lambda must be finite and nonnegative")
            if not math.isfinite(clip) or clip <= 0: raise ValueError("reward_clip must be finite and greater than zero")
            enabled=bool(cfg.get("enabled",self.enabled))
            ckpt=cfg.get("checkpoint")
            if self.enabled and enabled and self.fail and not ckpt: raise RewardModelCheckpointError(f"Missing reward checkpoint for {agent}")
            self.agents[agent]={"enabled":enabled,"lambda":lam,"reward_clip":clip,"checkpoint":ckpt,"validation":cfg.get("validation",r.get("validation",{})),"mean":float(cfg.get("reward_mean",0.0)),"std":float(cfg.get("reward_std",1.0)),"epsilon":float(cfg.get("epsilon",1e-6)),"checkpoint_hash":cfg.get("checkpoint_hash")}
        if self.enabled and self.fail:
            for agent,cfg in self.agents.items():
                if cfg["enabled"]: self._load(agent)
    def _load(self,agent:str)->RewardModelWrapper:
        if agent not in self.wrappers:
            cfg=self.agents[agent]
            self.wrappers[agent]=RewardModelWrapper(cfg.get("checkpoint"),enabled=cfg["enabled"],agent_type=agent,validation=cfg.get("validation"),fallback_to_env_reward=self.fallback,fail_on_invalid_reward_model=self.fail,reward_clip=None)
        return self.wrappers[agent]
    def score_transition(self,*,agent_type:str,event_type:str,environment_reward:float,state_features:Sequence[float],candidate_features:Sequence[float],selected_action_index:int,formal_mode:bool)->RewardContribution:
        try:
            event_type=validate_agent_event(agent_type,event_type)
        except Exception as exc:
            raise RewardModelCheckpointError(f"Invalid RLAIF event compatibility for agent={agent_type} event={event_type}: {exc}") from exc
        cfg=self.agents.get(agent_type)
        if not self.enabled or not cfg or not cfg.get("enabled",False):
            return RewardContribution(agent_type,event_type,float(environment_reward),0.0,0.0,0.0,0.0,0.0,float(environment_reward),False,None,0.0,1.0,0.0)
        if formal_mode and self.fallback:
            raise RewardModelCheckpointError("formal RLAIF mode cannot use fallback_to_env_reward")
        try:
            raw=float(self._load(agent_type).score(state_features,candidate_features,int(selected_action_index),event_type=event_type))
            if not math.isfinite(raw): raise RewardModelCheckpointError("Reward model produced non-finite output")
            return build_reward_contribution(agent_type=agent_type,event_type=event_type,environment_reward=float(environment_reward),raw_learned_reward=raw,mean=cfg["mean"],std=cfg["std"],epsilon=cfg["epsilon"],clip_bound=cfg["reward_clip"],lambda_value=cfg["lambda"],used_fallback=False,reward_checkpoint_hash=cfg.get("checkpoint_hash"))
        except Exception as exc:
            if formal_mode or self.fail or not self.fallback:
                raise RewardModelCheckpointError(f"RLAIF scoring failed for agent={agent_type} event={event_type} checkpoint={cfg.get('checkpoint')}: {exc}") from exc
            LOGGER.warning("RLAIF smoke fallback agent=%s event=%s checkpoint=%s reason=%s",agent_type,event_type,cfg.get("checkpoint"),exc)
            return RewardContribution(agent_type,event_type,float(environment_reward),0.0,0.0,0.0,cfg["lambda"],0.0,float(environment_reward),True,cfg.get("checkpoint_hash"),cfg["mean"],cfg["std"],cfg["reward_clip"])
    def score(self,agent:str,event_type:str,state_features:Sequence[float],candidate_features:Sequence[float],action_id:int|None=None)->float:
        return self.score_transition(agent_type=agent,event_type=event_type,environment_reward=0.0,state_features=state_features,candidate_features=candidate_features,selected_action_index=0 if action_id is None else action_id,formal_mode=self.fail).clipped_learned_reward
    def total_reward(self,agent:str,event_type:str,env_reward:float,state_features=None,candidate_features=None,action_id:int|None=None)->tuple[float,float]:
        c=self.score_transition(agent_type=agent,event_type=event_type,environment_reward=env_reward,state_features=state_features,candidate_features=candidate_features,selected_action_index=0 if action_id is None else action_id,formal_mode=self.fail)
        return c.total_reward,c.clipped_learned_reward

def empty_rlaif_training_totals()->dict[str,float|int]:
    d={}
    for a in AGENT_TYPES: d[f"rlaif_{a}_raw"]=0.0; d[f"rlaif_{a}_weighted"]=0.0
    d.update(rlaif_total_weighted=0.0, environment_reward_total=0.0, combined_reward_total=0.0, rlaif_fallback_count=0)
    return d

def update_rlaif_training_totals(totals:dict[str,float|int], contribution:RewardContribution)->dict[str,float|int]:
    totals[f"rlaif_{contribution.agent_type}_raw"] += contribution.raw_learned_reward
    totals[f"rlaif_{contribution.agent_type}_weighted"] += contribution.weighted_learned_contribution
    totals["environment_reward_total"] += contribution.environment_reward
    totals["rlaif_total_weighted"] = sum(float(totals[f"rlaif_{a}_weighted"]) for a in AGENT_TYPES)
    totals["combined_reward_total"] = float(totals["environment_reward_total"]) + float(totals["rlaif_total_weighted"])
    totals["rlaif_fallback_count"] += int(contribution.used_fallback)
    return totals
