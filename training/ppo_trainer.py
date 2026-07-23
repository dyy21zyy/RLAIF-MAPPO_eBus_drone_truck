"""Rollout collection, PPO updates, checkpointing, and evaluation for Stage 6."""

from __future__ import annotations

import csv
import json
import math
import hashlib
import subprocess
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from data_pipeline.build_instance import build_instance
from envs import DynamicDeliveryEnv
from envs.state_builder import action_name, build_candidate_action_features
from rlaif.preference_dataset import ACTION_FEATURE_KEYS
from training.assignment_ppo import AssignmentActorCritic
from training.bus_baseline_policy import BusBaselinePolicy, build_bus_baseline_policy
from training.ppo_buffer import AssignmentTransition, PPOBuffer
from training.reward_model_wrapper import RewardModelWrapper
from envs.status import is_delivered_status

METRIC_FIELDS = (
    "episode_reward", "episode_env_reward", "episode_rlaif_reward", "assignment_decision_count",
    "infeasible_action_count", "entropy", "policy_loss", "value_loss", "total_loss", "approx_kl",
    "clip_fraction", "explained_variance", "delivered_parcels", "undelivered_parcels", "average_lateness",
    "truck_direct_count", "truck_to_terminal_count", "truck_to_locker_count", "bus_charging_count",
)


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_environment(config: dict[str, Any], output_root: str | Path | None = None) -> DynamicDeliveryEnv:
    env_config = config["env"]
    instance_path = env_config.get("instance_path")
    if not instance_path and env_config.get("train_scenario_bank_manifest"):
        from evaluation.scenario_bank import load_scenario_bank
        bank = load_scenario_bank(env_config["train_scenario_bank_manifest"])
        if env_config.get("expected_train_bank_hash") and bank.bank_hash != env_config.get("expected_train_bank_hash"):
            raise ValueError("Assignment PPO train bank hash mismatch")
        instance_path = bank.scenarios[0].instance_path
    if instance_path:
        return DynamicDeliveryEnv(instance_path, env_config.get("config_path"))
    instance = build_instance(
        env_config["config_path"],
        fallback=bool(env_config.get("fallback", True)),
        output_root=output_root,
    )
    return DynamicDeliveryEnv(Path(instance["output_directory"]) / "instance.json", env_config.get("config_path"))


def _action_feature_vector(env: DynamicDeliveryEnv, action_id: int, feasible: bool) -> list[float]:
    parcel_id = env.current_decision.event.payload["parcel_id"]
    parcel = env.parcels[parcel_id]
    features = build_candidate_action_features(env, parcel, action_id, feasible)
    return [float(features[key]) for key in ACTION_FEATURE_KEYS]


def _episode_metrics(env: DynamicDeliveryEnv, counts: dict[str, int]) -> dict[str, float | int]:
    delivered = [parcel for parcel in env.parcels.values() if is_delivered_status(parcel.status)]
    lateness = [max(0.0, float(parcel.delivered_time_min) - parcel.deadline_min) for parcel in delivered]
    return {
        "delivered_parcels": len(delivered),
        "undelivered_parcels": len(env.parcels) - len(delivered),
        "average_lateness": float(np.mean(lateness)) if lateness else 0.0,
        "truck_direct_count": counts["TD"],
        "truck_to_terminal_count": counts["TBD"],
        "truck_to_locker_count": counts["TLD"],
        "bus_charging_count": counts["bus_charge"],
    }


def _fit_obs(values: list[float], dim: int) -> list[float]:
    vals = [float(v) for v in values]
    return vals[:dim] if len(vals) > dim else vals + [0.0] * (dim - len(vals))

def _first_feasible(action_mask: list[bool]) -> int:
    return next(index for index, feasible in enumerate(action_mask) if feasible)


def _is_bus_charge(observation: dict[str, Any], action: int) -> bool:
    if observation["agent"] != "bus" or observation.get("event_type") != "BUS_ARRIVAL":
        return False
    candidates = observation.get("candidate_actions", [])
    if candidates and 0 <= action < len(candidates):
        return candidates[action].get("action_type") == "charge"
    return False


def _non_assignment_action(
    env: DynamicDeliveryEnv,
    observation: dict[str, Any],
    bus_policy: BusBaselinePolicy,
) -> int:
    mask = [bool(value) for value in observation["action_mask"]]
    if observation["agent"] == "bus" and observation.get("event_type") == "BUS_ARRIVAL":
        return bus_policy.select_action(
            mask,
            env.config["bus"]["charging_actions_sec"],
            bus_soc=float(observation["features"][1]),
        )
    return _first_feasible(mask)


def collect_episode(
    env: DynamicDeliveryEnv,
    model: AssignmentActorCritic,
    buffer: PPOBuffer | None,
    bus_policy: BusBaselinePolicy,
    reward_wrapper: RewardModelWrapper,
    *,
    episode_id: int,
    lambda_rlaif: float,
    deterministic: bool = False,
) -> dict[str, float | int]:
    """Run one episode and store exactly one transition per assignment event."""
    observation, _ = env.reset(seed=episode_id)
    env_reward_total = 0.0
    rlaif_reward_total = 0.0
    assignment_total = 0.0
    all_zero_count = 0
    counts = {"TD": 0, "TBD": 0, "TLD": 0, "bus_charge": 0}

    while observation["agent"] != "terminal":
        if observation["agent"] != "assignment":
            action = _non_assignment_action(env, observation, bus_policy)
            if _is_bus_charge(observation, action):
                counts["bus_charge"] += 1
            observation, reward, *_ = env.step(action)
            env_reward_total += float(reward)
            continue

        obs = _fit_obs(observation["features"], model.obs_dim)
        mask = [bool(value) for value in observation["action_mask"]]
        parcel_id = str(observation["entity_id"])
        event_time = float(observation["time_min"])
        action, log_prob, value, used_fallback = model.act(obs, mask, deterministic=deterministic)
        all_zero_count += int(used_fallback)
        if not used_fallback and not mask[action]:
            raise RuntimeError("Masked PPO sampled an infeasible assignment action")
        chosen_name = action_name(env, action)
        counts[chosen_name.split("_", 1)[0]] += 1
        action_features = _action_feature_vector(env, action, mask[action] if mask else False)

        next_observation, reward, terminated, truncated, info = env.step(action)
        transition_env_reward = float(reward)
        env_reward_total += float(reward)
        while next_observation["agent"] not in {"assignment", "terminal"} and not (terminated or truncated):
            bus_action = _non_assignment_action(env, next_observation, bus_policy)
            if _is_bus_charge(next_observation, bus_action):
                counts["bus_charge"] += 1
            next_observation, bus_reward, terminated, truncated, info = env.step(bus_action)
            transition_env_reward += float(bus_reward)
            env_reward_total += float(bus_reward)

        r_rlaif = reward_wrapper.score(obs, action_features, action) if reward_wrapper.enabled else 0.0
        r_total = transition_env_reward + float(lambda_rlaif) * r_rlaif
        if not math.isfinite(r_total):
            raise RuntimeError("Assignment PPO received a non-finite reward")
        assignment_total += r_total
        rlaif_reward_total += r_rlaif
        done = bool(terminated or truncated)
        next_obs = (
            _fit_obs(next_observation["features"], model.obs_dim)
            if next_observation["agent"] == "assignment"
            else [0.0] * model.obs_dim
        )
        if buffer is not None:
            buffer.add(AssignmentTransition(
                obs=obs, action=action, action_mask=mask, log_prob=log_prob, value=value,
                reward=r_total, done=done, next_obs=next_obs,
                info={**info, "agent": "assignment", "all_zero_mask_fallback": used_fallback},
                episode_id=episode_id, event_time=event_time, parcel_id=parcel_id,
                chosen_action_name=chosen_name, r_env=transition_env_reward,
                r_rlaif=r_rlaif, r_total=r_total,
            ))
        observation = next_observation

    result: dict[str, float | int] = {
        "episode_reward": assignment_total,
        "episode_env_reward": env_reward_total,
        "episode_rlaif_reward": rlaif_reward_total,
        "assignment_decision_count": env.decision_counts["assignment"],
        "infeasible_action_count": env.infeasible_action_corrections + all_zero_count,
    }
    result.update(_episode_metrics(env, counts))
    return result


def update_ppo(
    model: AssignmentActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: PPOBuffer,
    training_config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, float]:
    buffer.compute_returns_and_advantages(
        gamma=float(training_config["gamma"]), gae_lambda=float(training_config["gae_lambda"])
    )
    advantages = buffer.advantages.copy()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    obs = torch.tensor(np.asarray([item.obs for item in buffer.transitions]), dtype=torch.float32)
    masks = torch.tensor(np.asarray([item.action_mask for item in buffer.transitions]), dtype=torch.bool)
    actions = torch.tensor([item.action for item in buffer.transitions], dtype=torch.long)
    old_log_probs = torch.tensor([item.log_prob for item in buffer.transitions], dtype=torch.float32)
    returns = torch.tensor(buffer.returns, dtype=torch.float32)
    normalized_advantages = torch.tensor(advantages, dtype=torch.float32)
    aggregate: dict[str, list[float]] = {key: [] for key in (
        "entropy", "policy_loss", "value_loss", "total_loss", "approx_kl", "clip_fraction"
    )}
    for _ in range(int(training_config["ppo_epochs"])):
        for indices in buffer.minibatch_indices(int(training_config["batch_size"]), rng):
            index = torch.as_tensor(indices, dtype=torch.long)
            new_log_prob, entropy, values = model.evaluate_actions(obs[index], actions[index], masks[index])
            log_ratio = new_log_prob - old_log_probs[index]
            ratio = log_ratio.exp()
            unclipped = ratio * normalized_advantages[index]
            clipped = ratio.clamp(1.0 - float(training_config["clip_eps"]), 1.0 + float(training_config["clip_eps"])) * normalized_advantages[index]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = nn.functional.mse_loss(values, returns[index])
            entropy_mean = entropy.mean()
            total_loss = policy_loss + float(training_config["vf_coef"]) * value_loss - float(training_config["ent_coef"]) * entropy_mean
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(training_config["max_grad_norm"]))
            optimizer.step()
            values_to_add = {
                "entropy": entropy_mean, "policy_loss": policy_loss, "value_loss": value_loss,
                "total_loss": total_loss, "approx_kl": ((ratio - 1.0) - log_ratio).mean(),
                "clip_fraction": ((ratio - 1.0).abs() > float(training_config["clip_eps"])).float().mean(),
            }
            for key, metric in values_to_add.items():
                aggregate[key].append(float(metric.detach()))
    predictions = np.asarray([item.value for item in buffer.transitions], dtype=np.float32)
    variance = float(np.var(buffer.returns))
    explained_variance = 0.0 if variance < 1e-12 else 1.0 - float(np.var(buffer.returns - predictions)) / variance
    result = {key: float(np.mean(values)) for key, values in aggregate.items()}
    result["explained_variance"] = explained_variance
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("PPO update produced non-finite metrics")
    return result


def _file_hash(path: str | Path | None) -> str | None:
    if not path or not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _assignment_lineage(config: dict[str, Any]) -> dict[str, Any]:
    env = config.get("env", {})
    return {
        "algorithm_identity": "assignment_ppo",
        "training_seed": config.get("training", {}).get("seed"),
        "train_bank_path": env.get("train_scenario_bank_manifest"),
        "train_bank_hash": env.get("expected_train_bank_hash"),
        "validation_bank_path": env.get("validation_scenario_bank_manifest"),
        "validation_bank_hash": env.get("expected_validation_bank_hash"),
        "environment_config_hash": _file_hash(env.get("config_path")),
        "policy_architecture": config.get("policy", {}),
        "training_hyperparameters": config.get("training", {}),
        "fixed_baseline_policy_identities": config.get("fixed_baseline_policies", {"truck": "first_feasible", "bus": config.get("bus_baseline", {}).get("name"), "station": "first_feasible"}),
        "code_commit": _code_commit(),
    }


def save_assignment_checkpoint(path: str | Path, model: AssignmentActorCritic, optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "obs_dim": model.obs_dim, "action_dim": model.action_dim, "hidden_dims": list(model.hidden_dims),
        "config": config, "stage": 6, "agent": "assignment",
        "algorithm": "assignment_ppo", "lineage": _assignment_lineage(config),
    }, path)


def load_assignment_checkpoint(path: str | Path) -> tuple[AssignmentActorCritic, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != 6 or checkpoint.get("agent") != "assignment":
        raise ValueError("Checkpoint is not a Stage 6 assignment PPO checkpoint")
    model = AssignmentActorCritic(
        int(checkpoint["obs_dim"]), int(checkpoint["action_dim"]), checkpoint["hidden_dims"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", *METRIC_FIELDS])
        writer.writeheader()
        writer.writerows(rows)


def train_assignment_ppo(config: dict[str, Any], *, output_root: str | Path | None = None) -> dict[str, Any]:
    training = config["training"]
    seed = int(training["seed"])
    set_training_seed(seed)
    env = create_environment(config, output_root=output_root)
    observation, _ = env.reset(seed=seed)
    model = AssignmentActorCritic(len(observation["features"]), env.assignment_action_size, config["policy"]["hidden_dims"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["lr"]))
    bus_policy = build_bus_baseline_policy(config["bus_baseline"])
    reward_wrapper = RewardModelWrapper(
        config["rlaif"].get("reward_model_checkpoint"), enabled=bool(config["rlaif"].get("enabled", False))
    )
    buffer = PPOBuffer()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    total_episodes = int(training["total_episodes"])
    rollout_episodes = int(training["rollout_episodes"])
    for start in range(0, total_episodes, rollout_episodes):
        rollout_metrics = []
        for episode in range(start, min(start + rollout_episodes, total_episodes)):
            rollout_metrics.append(collect_episode(
                env, model, buffer, bus_policy, reward_wrapper, episode_id=seed + episode,
                lambda_rlaif=float(config["rlaif"].get("lambda_rlaif", 1.0)),
            ))
        update_metrics = update_ppo(model, optimizer, buffer, training, rng)
        for offset, episode_metrics in enumerate(rollout_metrics):
            row = {"episode": start + offset + 1, **episode_metrics, **update_metrics}
            rows.append({key: row[key] for key in ("episode", *METRIC_FIELDS)})
        buffer.clear()
    save_assignment_checkpoint(config["output"]["checkpoint_path"], model, optimizer, config)
    _write_csv(config["output"]["training_log_path"], rows)
    return {"model": model, "rows": rows, "checkpoint_path": config["output"]["checkpoint_path"]}


def evaluate_assignment_ppo(
    config: dict[str, Any], checkpoint_path: str | Path, *, output_root: str | Path | None = None,
    episodes: int = 1,
) -> dict[str, Any]:
    env = create_environment(config, output_root=output_root)
    model, _ = load_assignment_checkpoint(checkpoint_path)
    if model.action_dim != env.assignment_action_size:
        raise ValueError("Checkpoint assignment action dimension does not match the environment")
    bus_policy = build_bus_baseline_policy(config["bus_baseline"])
    reward_wrapper = RewardModelWrapper(
        config["rlaif"].get("reward_model_checkpoint"), enabled=bool(config["rlaif"].get("enabled", False))
    )
    results = [collect_episode(
        env, model, None, bus_policy, reward_wrapper, episode_id=int(config["training"]["seed"]) + index,
        lambda_rlaif=float(config["rlaif"].get("lambda_rlaif", 1.0)), deterministic=True,
    ) for index in range(episodes)]
    summary = {key: float(np.mean([float(row[key]) for row in results])) for key in results[0]}
    payload = {"episodes": results, "summary": summary, "checkpoint": str(checkpoint_path)}
    path = Path(config["output"]["eval_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
