"""Structured per-agent RLAIF reward accounting."""
from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class RewardContribution:
    agent_type: str
    event_type: str
    environment_reward: float
    raw_learned_reward: float
    normalized_learned_reward: float
    clipped_learned_reward: float
    lambda_value: float
    weighted_learned_contribution: float
    total_reward: float
    used_fallback: bool
    reward_checkpoint_hash: str | None = None
    normalization_mean: float = 0.0
    normalization_std: float = 1.0
    clip_bound: float = 0.0
    def __post_init__(self):
        for name in ("environment_reward","raw_learned_reward","normalized_learned_reward","clipped_learned_reward","lambda_value","weighted_learned_contribution","total_reward","normalization_mean","normalization_std","clip_bound"):
            value=float(getattr(self,name))
            if not math.isfinite(value):
                raise ValueError(f"RewardContribution {name} must be finite")
        if abs(float(self.total_reward) - (float(self.environment_reward)+float(self.weighted_learned_contribution))) > 1e-6:
            raise ValueError("RewardContribution total_reward must equal environment plus weighted learned contribution")

def build_reward_contribution(*, agent_type: str, event_type: str, environment_reward: float, raw_learned_reward: float, mean: float, std: float, epsilon: float=1e-6, clip_bound: float, lambda_value: float, used_fallback: bool=False, reward_checkpoint_hash: str|None=None) -> RewardContribution:
    if lambda_value < 0 or not math.isfinite(lambda_value): raise ValueError("lambda must be finite and nonnegative")
    if clip_bound <= 0 or not math.isfinite(clip_bound): raise ValueError("reward_clip must be finite and positive")
    normalized=(float(raw_learned_reward)-float(mean))/(float(std)+float(epsilon))
    clipped=max(-float(clip_bound), min(float(clip_bound), normalized))
    weighted=float(lambda_value)*clipped
    return RewardContribution(agent_type,event_type,float(environment_reward),float(raw_learned_reward),normalized,clipped,float(lambda_value),weighted,float(environment_reward)+weighted,bool(used_fallback),reward_checkpoint_hash,float(mean),float(std),float(clip_bound))
