from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence
from training.reward_model_wrapper import RewardModelCheckpointError, RewardModelWrapper

class RewardRegistry:
    def __init__(self, config:dict[str,Any]):
        r=config.get("rlaif",config); self.enabled=bool(r.get("enabled",False)); self.fallback=bool(r.get("fallback_to_env_reward",False)); self.fail=bool(r.get("fail_on_invalid_reward_model",True)); self.agents={}; self.wrappers={}
        for agent,cfg in r.get("agents",{}).items():
            self.agents[agent]={"enabled":bool(cfg.get("enabled",self.enabled)),"lambda":float(cfg.get("lambda",0.0)),"reward_clip":float(cfg.get("reward_clip",2.0)),"checkpoint":cfg.get("checkpoint"),"validation":cfg.get("validation",r.get("validation",{}))}
        if self.enabled and self.fail:
            for agent,cfg in self.agents.items():
                if cfg["enabled"]: self._load(agent)
    def _load(self,agent:str)->RewardModelWrapper:
        if agent not in self.wrappers:
            cfg=self.agents[agent]
            self.wrappers[agent]=RewardModelWrapper(cfg.get("checkpoint"),enabled=cfg["enabled"],agent_type=agent,validation=cfg.get("validation"),fallback_to_env_reward=self.fallback,fail_on_invalid_reward_model=self.fail,reward_clip=cfg.get("reward_clip"))
        return self.wrappers[agent]
    def score(self,agent:str,event_type:str,state_features:Sequence[float],candidate_features:Sequence[float],action_id:int|None=None)->float:
        cfg=self.agents.get(agent)
        if not self.enabled or not cfg or not cfg.get("enabled",False): return 0.0
        return self._load(agent).score(state_features,candidate_features,0 if action_id is None else action_id,event_type=event_type)
    def total_reward(self,agent:str,event_type:str,env_reward:float,state_features=None,candidate_features=None,action_id:int|None=None)->tuple[float,float]:
        cfg=self.agents.get(agent)
        if not self.enabled or not cfg or not cfg.get("enabled",False): return float(env_reward),0.0
        if state_features is None or candidate_features is None: raise ValueError("Enabled RLAIF requires state and candidate features")
        learned=self.score(agent,event_type,state_features,candidate_features,action_id)
        return float(env_reward)+float(cfg["lambda"])*learned, learned
