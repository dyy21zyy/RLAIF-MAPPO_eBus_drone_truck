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
from rlaif.reward_registry import RewardRegistry

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

def _score_selected(
    reward_registry,
    observation,
    action,
    reward,
    info,
    rlaif,
):
    if reward_registry is None:
        return

    agent = str(
        observation.get("agent_id")
    )

    if agent not in (
        "assignment",
        "truck",
        "bus",
        "station",
    ):
        return

    if isinstance(
        reward_registry,
        RewardRegistry,
    ):
        candidate_rows = (
            observation.get(
                "candidate_features"
            )
            or []
        )

        selected_action = int(action)

        if (
            selected_action < 0
            or selected_action
            >= len(candidate_rows)
        ):
            raise ValueError(
                "Selected action has no matching "
                "candidate feature row"
            )

        event_type = (
            observation.get(
                "event_type_detail"
            )
            or observation.get(
                "event_type"
            )
        )

        contribution = (
            reward_registry.score_transition(
                agent_type=agent,
                event_type=event_type,
                environment_reward=float(
                    reward
                ),
                state_features=[
                    float(value)
                    for value in (
                        observation.get(
                            "features"
                        )
                        or []
                    )
                ],
                candidate_features=[
                    float(value)
                    for value in (
                        candidate_rows[
                            selected_action
                        ]
                    )
                ],
                selected_action_index=(
                    selected_action
                ),
                formal_mode=(
                    reward_registry.formal_mode
                ),
            )
        )

        out = {
            "raw": (
                contribution
                .raw_learned_reward
            ),
            "normalized": (
                contribution
                .normalized_learned_reward
            ),
            "clipped": (
                contribution
                .clipped_learned_reward
            ),
            "weighted": (
                contribution
                .weighted_learned_contribution
            ),
            "fallback": (
                contribution.used_fallback
            ),
        }

    elif hasattr(
        reward_registry,
        "score_transition",
    ):
        # Compatibility path for lightweight
        # diagnostic registries used by tests.
        out = reward_registry.score_transition(
            agent=agent,
            event_type=observation.get(
                "event_type"
            ),
            observation=observation,
            action=action,
            environment_reward=reward,
            info=info,
        )

    elif hasattr(
        reward_registry,
        "score",
    ):
        out = reward_registry.score(
            agent,
            observation,
            action,
            info,
        )

    else:
        out = {
            "raw": 0.0,
            "normalized": 0.0,
            "clipped": 0.0,
            "weighted": 0.0,
            "fallback": False,
        }

    for key in (
        "raw",
        "normalized",
        "clipped",
        "weighted",
    ):
        rlaif[
            f"rlaif_{agent}_{key}"
        ] += float(
            out.get(
                key,
                out.get(
                    f"{key}_reward",
                    0.0,
                ),
            )
        )

    if out.get("fallback"):
        rlaif[
            "rlaif_fallback_count"
        ] += 1.0

    rlaif[
        "rlaif_total_weighted"
    ] = sum(
        rlaif[
            f"rlaif_{name}_weighted"
        ]
        for name in (
            "assignment",
            "truck",
            "bus",
            "station",
        )
    )

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
        invariant_errors = env.check_invariants()
        if invariant_errors:
            raise RuntimeError("hard locker invariant failure: " + "; ".join(invariant_errors))
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
