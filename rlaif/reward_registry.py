from __future__ import annotations
import logging, math
from typing import Any, Sequence
from training.event_schema import AGENT_TYPES, REQUIRED_EVENT_COVERAGE, validate_agent_event
from training.reward_contribution import RewardContribution
from training.reward_model_wrapper import RewardModelCheckpointError
from rlaif.runtime_agent_reward_model import RuntimeAgentRewardModel

LOGGER=logging.getLogger(__name__)
FULL=set(AGENT_TYPES)

class RewardRegistry:
    def __init__(self, config:dict[str,Any]):
        r=config.get("rlaif",config)
        self.enabled=bool(r.get("enabled",False))
        self.scope=str(r.get("scope", "none" if not self.enabled else "assignment"))
        self.formal_mode=str(config.get("run_classification", r.get("run_classification", "formal")))=="formal"
        self.fallback=bool(r.get("fallback_to_env_reward",False))
        self.fail=bool(r.get("fail_on_invalid_reward_model",True))
        if self.formal_mode and self.enabled and (self.fallback or not self.fail):
            raise ValueError("formal RLAIF requires fallback_to_env_reward=false and fail_on_invalid_reward_model=true")
        self.agents:dict[str,dict[str,Any]]={}
        self.models:dict[str,RuntimeAgentRewardModel]={}
        agent_cfg=r.get("agents",{})
        for agent in AGENT_TYPES:
            cfg=dict(agent_cfg.get(agent, {}))
            enabled=bool(cfg.get("enabled", self.enabled and ((self.scope=="all") or (self.scope==agent=="assignment"))))
            lam=float(cfg.get("lambda", r.get("lambda", 1.0)))
            clip=float(cfg.get("reward_clip", r.get("reward_clip", 2.0)))
            if not math.isfinite(lam) or lam < 0: raise ValueError("lambda must be finite and nonnegative")
            if not math.isfinite(clip) or clip <= 0: raise ValueError("reward_clip must be finite and greater than zero")
            self.agents[agent]={"enabled":enabled,"lambda":lam,"reward_clip":clip,"checkpoint":cfg.get("checkpoint"),"checkpoint_hash":cfg.get("checkpoint_hash")}
        enabled={a for a,c in self.agents.items() if c["enabled"]}
        if not self.enabled and enabled:
            raise ValueError("RLAIF disabled but reward agents are enabled")
        if self.formal_mode and self.enabled:
            if self.scope=="assignment" and enabled != {"assignment"}: raise ValueError("formal assignment RLAIF requires exactly the assignment reward model")
            if self.scope=="all" and enabled != FULL: raise ValueError("formal full RLAIF requires assignment, truck, bus, and station reward models")
            if self.scope not in {"assignment","all"}: raise ValueError(f"unsupported formal RLAIF scope: {self.scope}")
        if self.enabled and self.fail:
            for a in enabled: self._load(a)

    def _load(self,agent:str)->RuntimeAgentRewardModel:
        if agent not in self.models:
            cfg=self.agents[agent]
            if not cfg.get("checkpoint"):
                raise RewardModelCheckpointError(f"Missing reward checkpoint for {agent}")
            self.models[agent]=RuntimeAgentRewardModel.from_checkpoint(
                cfg["checkpoint"], expected_agent_type=agent, expected_event_types=sorted(REQUIRED_EVENT_COVERAGE[agent]),
                expected_checkpoint_hash=cfg.get("checkpoint_hash"), formal_mode=self.formal_mode)
        return self.models[agent]

    def score_transition(self,*,agent_type:str,event_type:str,environment_reward:float,state_features:Sequence[float],candidate_features:Sequence[float],selected_action_index:int=0,formal_mode:bool|None=None)->RewardContribution:
        event_type=validate_agent_event(agent_type,event_type)
        formal = self.formal_mode if formal_mode is None else bool(formal_mode)
        cfg=self.agents.get(agent_type)
        if not self.enabled or not cfg or not cfg.get("enabled",False):
            return RewardContribution(agent_type,event_type,float(environment_reward),0.0,0.0,0.0,0.0,0.0,float(environment_reward),False,None)
        if formal and self.fallback:
            raise RewardModelCheckpointError("formal RLAIF mode cannot use fallback_to_env_reward")
        try:
            score=self._load(agent_type).score(state_features=state_features,candidate_features=candidate_features,event_type=event_type)
            clip=float(cfg["reward_clip"]); clipped=max(-clip,min(clip,score.normalized_score)); weighted=float(cfg["lambda"])*clipped
            return RewardContribution(agent_type,event_type,float(environment_reward),score.raw_score,score.normalized_score,clipped,float(cfg["lambda"]),weighted,float(environment_reward)+weighted,False,self.models[agent_type].checkpoint_hash)
        except Exception as exc:
            if formal or self.fail or not self.fallback:
                raise RewardModelCheckpointError(f"RLAIF scoring failed for agent={agent_type} event={event_type} checkpoint={cfg.get('checkpoint')}: {exc}") from exc
            LOGGER.warning("RLAIF diagnostic fallback agent=%s event=%s checkpoint=%s reason=%s",agent_type,event_type,cfg.get("checkpoint"),exc)
            return RewardContribution(agent_type,event_type,float(environment_reward),0.0,0.0,0.0,float(cfg["lambda"]),0.0,float(environment_reward),True,cfg.get("checkpoint_hash"))

    def score(self,agent:str,event_type:str,state_features:Sequence[float],candidate_features:Sequence[float],action_id:int|None=None)->float:
        return self.score_transition(agent_type=agent,event_type=event_type,environment_reward=0.0,state_features=state_features,candidate_features=candidate_features,selected_action_index=0 if action_id is None else action_id).clipped_learned_reward

    def total_reward(self,agent:str,event_type:str,env_reward:float,state_features=None,candidate_features=None,action_id:int|None=None)->tuple[float,float]:
        c=self.score_transition(agent_type=agent,event_type=event_type,environment_reward=env_reward,state_features=state_features,candidate_features=candidate_features,selected_action_index=0 if action_id is None else action_id)
        return c.total_reward,c.clipped_learned_reward

    def lineage(self)->dict[str,Any]:
        enabled=[a for a,c in self.agents.items() if c.get("enabled")]
        return {"rlaif_scope": self.scope if self.enabled else "none", "enabled_reward_agents": enabled,
                "reward_checkpoint_paths": {a:str(self.agents[a]["checkpoint"]) for a in enabled},
                "reward_checkpoint_hashes": {a:self.models[a].checkpoint_hash for a in enabled if a in self.models},
                "reward_lambda_by_agent": {a:self.agents[a]["lambda"] for a in enabled},
                "reward_clip_by_agent": {a:self.agents[a]["reward_clip"] for a in enabled}}

def empty_rlaif_training_totals()->dict[str,float|int]:
    d={}
    for a in AGENT_TYPES:
        for stage in ("raw","normalized","clipped","weighted"):
            d[f"rlaif_{a}_{stage}"]=0.0
    d.update(rlaif_total_weighted=0.0, environment_reward_total=0.0, combined_reward_total=0.0, rlaif_fallback_count=0)
    return d

def update_rlaif_training_totals(totals:dict[str,float|int], contribution:RewardContribution)->dict[str,float|int]:
    a=contribution.agent_type
    totals[f"rlaif_{a}_raw"] += contribution.raw_learned_reward
    totals[f"rlaif_{a}_normalized"] += contribution.normalized_learned_reward
    totals[f"rlaif_{a}_clipped"] += contribution.clipped_learned_reward
    totals[f"rlaif_{a}_weighted"] += contribution.weighted_learned_contribution
    totals["environment_reward_total"] += contribution.environment_reward
    totals["rlaif_total_weighted"] = sum(float(totals[f"rlaif_{agent}_weighted"]) for agent in AGENT_TYPES)
    totals["combined_reward_total"] = float(totals["environment_reward_total"]) + float(totals["rlaif_total_weighted"])
    totals["rlaif_fallback_count"] += int(contribution.used_fallback)
    return totals
