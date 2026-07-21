from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any
import math

class TrainingConfigError(ValueError):
    """Raised when a MAPPO training configuration is incomplete or inconsistent."""

REWARD_COMPONENTS = ["passenger_delay","bus_operating_delay","parcel_lateness","energy_cost","power_overload","bus_battery_violation","locker_overflow","truck_cost","undelivered","battery_shortage","infeasible_action"]

def _need(d, path):
    cur=d
    for p in path.split('.'):
        if not isinstance(cur, dict) or p not in cur:
            raise TrainingConfigError(f"missing required config field: {path}")
        cur=cur[p]
    return cur

def _alias(section, canonical, alias):
    if canonical in section and alias in section and section[canonical] != section[alias]:
        raise TrainingConfigError(f"conflicting {canonical} and {alias}")
    if canonical not in section and alias in section:
        section[canonical] = section[alias]
    section.pop(alias, None)

def _positive(name, value, *, allow_zero=False, max_one=False):
    if isinstance(value, bool): raise TrainingConfigError(f"{name} must be numeric")
    f=float(value)
    if not math.isfinite(f) or (f < 0 if allow_zero else f <= 0) or (max_one and f > 1):
        raise TrainingConfigError(f"invalid {name}: {value}")
    return f

def validate_nonzero_formal_reward(config: dict[str, Any]) -> None:
    reward = config.get("reward")
    if not isinstance(reward, dict):
        raise TrainingConfigError("formal config missing reward block")
    weights=[]
    for name in REWARD_COMPONENTS:
        if name not in reward:
            raise TrainingConfigError(f"formal reward missing component: {name}")
        v=reward[name]
        if isinstance(v, bool): raise TrainingConfigError(f"reward {name} must not be boolean")
        f=float(v)
        if not math.isfinite(f) or f < 0:
            raise TrainingConfigError(f"reward {name} must be finite and non-negative")
        weights.append(f)
    if all(v == 0 for v in weights):
        raise TrainingConfigError("formal reward block cannot be all zero")
    if reward.get("apply_reference_scales", False) and not reward.get("scale_artifact"):
        raise TrainingConfigError("reference scaling enabled without scale_artifact")

def resolve_mappo_training_config(config: dict[str, Any], *, seed_override: int | None = None, mode_override: str | None = None, output_root_override: str | Path | None = None) -> dict[str, Any]:
    cfg=deepcopy(config)
    mode = mode_override or cfg.get("mode", "environment_reward")
    env=dict(_need(cfg,"env")); tr=dict(_need(cfg,"training")); net=dict(_need(cfg,"networks")); rlaif=dict(cfg.get("rlaif", {})); out=dict(_need(cfg,"output"))
    _alias(tr,"entropy_coef","ent_coef"); _alias(tr,"value_coef","vf_coef")
    seed = seed_override if seed_override is not None else tr.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int): raise TrainingConfigError("training.seed must be an integer")
    tr["seed"] = seed; tr.pop("training_seeds", None)
    for k in ["total_episodes","rollout_episodes","ppo_epochs","batch_size"]:
        tr[k]=int(_positive(f"training.{k}", _need(tr,k)))
    if tr["rollout_episodes"] > tr["total_episodes"]: raise TrainingConfigError("rollout_episodes must be <= total_episodes")
    for k in ["lr_actor","lr_critic","clip_eps","max_grad_norm","event_time_reference_min"]: tr[k]=_positive(f"training.{k}", _need(tr,k))
    tr["gamma"]=_positive("training.gamma", _need(tr,"gamma"), max_one=True)
    tr["gae_lambda"]=_positive("training.gae_lambda", _need(tr,"gae_lambda"), allow_zero=True, max_one=True)
    tr["entropy_coef"]=_positive("training.entropy_coef", _need(tr,"entropy_coef"), allow_zero=True)
    tr["value_coef"]=_positive("training.value_coef", _need(tr,"value_coef"), allow_zero=True)
    tr["optimizer"] = str(_need(tr,"optimizer"))
    for k in ["assignment_hidden_dims","truck_hidden_dims","bus_hidden_dims","station_hidden_dims","critic_hidden_dims"]:
        if not isinstance(_need(net,k), list) or not net[k]: raise TrainingConfigError(f"networks.{k} must be non-empty list")
    rlaif.setdefault("enabled", mode == "rlaif_reward"); rlaif.setdefault("fallback_to_env_reward", False); rlaif.setdefault("fail_on_invalid_reward_model", True)
    root = Path(output_root_override or _need(out,"output_root"))
    def path_for(template_key, path_key):
        if path_key in out: return str(out[path_key])
        return str(root / str(_need(out, template_key)).format(seed=seed))
    out["output_root"] = str(root)
    out["checkpoint_path"] = path_for("checkpoint_name_template","checkpoint_path")
    out["training_log_path"] = path_for("training_log_name_template","training_log_path")
    out["eval_path"] = path_for("eval_name_template","eval_path")
    out["resolved_config_path"] = path_for("resolved_config_name_template","resolved_config_path")
    if "reward" in cfg: validate_nonzero_formal_reward(cfg)
    return {"mode": mode, "env": {"config_path": str(_need(env,"config_path")), "fallback": bool(env.get("fallback", False))}, "training": tr, "networks": net, "rlaif": rlaif, "output": out, "reward": cfg.get("reward", {})}

def validate_reward_artifacts(config: dict[str, Any], *, config_only: bool) -> None:
    if config_only: return
    r=config.get("rlaif", {})
    if r.get("enabled"):
        for name, agent in r.get("agents", {}).items():
            if agent.get("enabled", True) and not Path(agent.get("checkpoint", "")).exists():
                raise TrainingConfigError(f"missing RLAIF reward checkpoint for {name}: {agent.get('checkpoint')}")
