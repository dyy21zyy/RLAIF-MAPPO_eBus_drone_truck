"""Canonical real frozen-scenario episode runner."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import time
from pathlib import Path

from envs import DynamicDeliveryEnv
from evaluation.scenario_bank import load_frozen_instance
from evaluation.formal_metrics import collect_formal_metrics
from evaluation.formal_metric_validation import validate_formal_metrics

@dataclass(frozen=True)
class FormalEpisodeResult:
    status: str
    metrics: dict[str, Any]
    metric_sources: dict[str, Any]
    rlaif_decomposition: dict[str, float]
    runtime_seconds: float
    transition_count: int
    failure_reason: str | None
    exception_type: str | None

def _zero_rlaif() -> dict[str, float]:
    d={}
    for a in ("assignment","truck","bus","station"):
        for k in ("raw","normalized","clipped","weighted"):
            d[f"rlaif_{a}_{k}"]=0.0
    d.update(rlaif_total_weighted=0.0, rlaif_fallback_count=0.0)
    return d

def _score_selected(reward_registry, observation, action, reward, info, rlaif):
    if reward_registry is None: return
    agent=str(observation.get("agent_id"))
    if agent not in ("assignment","truck","bus","station"): return
    if hasattr(reward_registry, "score_transition"):
        out=reward_registry.score_transition(agent=agent,event_type=observation.get("event_type"),observation=observation,action=action,environment_reward=reward,info=info)
    elif hasattr(reward_registry, "score"):
        out=reward_registry.score(agent, observation, action, info)
    else:
        out={"raw":0.0,"normalized":0.0,"clipped":0.0,"weighted":0.0,"fallback":False}
    for k in ("raw","normalized","clipped","weighted"):
        rlaif[f"rlaif_{agent}_{k}"] += float(out.get(k, out.get(f"{k}_reward", 0.0)))
    if out.get("fallback"): rlaif["rlaif_fallback_count"] += 1.0
    rlaif["rlaif_total_weighted"] = sum(rlaif[f"rlaif_{a}_weighted"] for a in ("assignment","truck","bus","station"))

def evaluate_policy_on_frozen_scenario(*, scenario, method_spec, policy, reward_registry, evaluation_config, training_seed) -> FormalEpisodeResult:
    started=time.perf_counter(); rlaif=_zero_rlaif(); transitions=0
    try:
        inst=load_frozen_instance(scenario)
        env=DynamicDeliveryEnv(Path(scenario.instance_path))
        obs,_=env.reset(seed=training_seed)
        limit=int(evaluation_config.get("max_decisions", 10000)) if isinstance(evaluation_config, dict) else 10000
        while obs.get("agent_id") != "terminal" and transitions < limit:
            action=policy.select_action(observation=obs, env=env, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            transitions += 1
            _score_selected(reward_registry, obs, action, reward, info, rlaif)
            obs = next_obs
            if terminated or truncated: break
        runtime=time.perf_counter()-started
        if transitions <= 0: raise RuntimeError("successful rollout requires env.step transition_count > 0")
        metrics, sources = collect_formal_metrics(env, runtime_seconds=runtime, transition_count=transitions, rlaif=rlaif)
        flat={k:v for k,v in metrics.items()}
        validate_formal_metrics(flat)
        return FormalEpisodeResult("success", metrics, sources, rlaif, runtime, transitions, None, None)
    except Exception as exc:
        runtime=time.perf_counter()-started
        status="failed_environment_runtime"
        name=type(exc).__name__
        if "Metric" in name or "metric" in str(exc): status="failed_metric_validation"
        if "scenario" in str(exc).lower() or isinstance(exc, (FileNotFoundError, ValueError)): status="failed_scenario_validation"
        return FormalEpisodeResult(status, {}, {}, rlaif, runtime, transitions, str(exc), name)
