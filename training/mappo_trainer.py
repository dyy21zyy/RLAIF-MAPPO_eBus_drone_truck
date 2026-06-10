"""Rollout collection, PPO updates, checkpoints, and evaluation for Stage 7."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from envs.state_builder import build_candidate_action_features
from rlaif.preference_dataset import ACTION_FEATURE_KEYS
from training.mappo_async import reward_decomposition, transition_reward, validate_decision
from training.mappo_buffer import AsyncMAPPOBuffer, AsyncTransition
from training.mappo_networks import AssignmentActor, BusActor, CentralizedCritic, BUS_CHARGING_ACTIONS
from training.ppo_trainer import create_environment
from training.reward_model_wrapper import RewardModelWrapper

FEATURE_SCHEMA_VERSION = 1
METRIC_FIELDS = (
    "episode_reward", "episode_env_reward", "episode_rlaif_reward", "assignment_decision_count",
    "bus_decision_count", "assignment_policy_loss", "bus_policy_loss", "value_loss",
    "entropy_assignment", "entropy_bus", "approx_kl_assignment", "approx_kl_bus",
    "clip_fraction_assignment", "clip_fraction_bus", "delivered_parcels", "undelivered_parcels",
    "average_lateness", "infeasible_action_count", "bus_charging_count", "passenger_delay",
    "bus_operating_delay", "power_overload_amount", "locker_overflow_amount",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _assignment_action_features(env, action: int, feasible: bool) -> list[float]:
    parcel = env.parcels[env.current_decision.event.payload["parcel_id"]]
    features = build_candidate_action_features(env, parcel, action, feasible)
    return [float(features[key]) for key in ACTION_FEATURE_KEYS]


def _episode_summary(env, env_reward: float, rlaif_reward: float, bus_charging_count: int) -> dict[str, float | int]:
    delivered = [parcel for parcel in env.parcels.values() if parcel.status == "delivered"]
    lateness = [max(0.0, float(parcel.delivered_time_min) - parcel.deadline_min) for parcel in delivered]
    costs = env.cost_components
    return {
        "episode_reward": env_reward + rlaif_reward,
        "episode_env_reward": env_reward,
        "episode_rlaif_reward": rlaif_reward,
        "assignment_decision_count": env.decision_counts["assignment"],
        "bus_decision_count": env.decision_counts["bus"],
        "delivered_parcels": len(delivered),
        "undelivered_parcels": len(env.parcels) - len(delivered),
        "average_lateness": float(np.mean(lateness)) if lateness else 0.0,
        "infeasible_action_count": env.infeasible_action_corrections,
        "bus_charging_count": bus_charging_count,
        "passenger_delay": float(costs.get("passenger_delay", 0.0)),
        "bus_operating_delay": float(costs.get("bus_operating_delay", 0.0)),
        "power_overload_amount": float(costs.get("power_overload", 0.0)),
        "locker_overflow_amount": float(costs.get("locker_overflow", 0.0)),
    }


def collect_episode(
    env, assignment_actor: AssignmentActor, bus_actor: BusActor, critic: CentralizedCritic,
    buffer: AsyncMAPPOBuffer | None, reward_wrapper: RewardModelWrapper, *, episode_id: int,
    lambda_rlaif: float, deterministic: bool = False,
) -> dict[str, float | int]:
    """Act once and append once for each actual decision event."""
    observation, _ = env.reset(seed=episode_id)
    env_total = rlaif_total = 0.0
    bus_charging_count = 0
    while observation["agent_id"] != "terminal":
        agent_id = str(observation["agent_id"])
        event_type = str(observation["event_type"])
        validate_decision(agent_id, event_type)
        local_obs = [float(value) for value in observation["features"]]
        mask = [bool(value) for value in observation["action_mask"]]
        global_state = [float(value) for value in env.get_global_state()]
        actor = assignment_actor if agent_id == "assignment" else bus_actor
        action, log_prob = actor.act(local_obs, mask, deterministic=deterministic)
        if not mask[action]:
            raise RuntimeError("Masked MAPPO sampled an infeasible action")
        with torch.no_grad():
            value = float(critic(torch.tensor(global_state, dtype=torch.float32)).item())
        action_features = None
        if agent_id == "assignment":
            action_features = _assignment_action_features(env, action, mask[action])
        elif BUS_CHARGING_ACTIONS[action] > 0:
            bus_charging_count += 1
        next_observation, env_reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        total_reward, learned_reward = transition_reward(
            agent_id, float(env_reward), reward_wrapper, lambda_rlaif=lambda_rlaif,
            state_features=local_obs, action_features=action_features, action_id=action,
        )
        env_total += float(env_reward)
        rlaif_total += learned_reward * float(lambda_rlaif)
        if buffer is not None:
            buffer.append(AsyncTransition(
                agent_id=agent_id, local_obs=local_obs, global_state=global_state, action=action,
                action_mask=mask, log_prob=log_prob, value=value, reward=total_reward, done=done,
                next_global_state=[float(value) for value in env.get_global_state()],
                event_type=event_type, event_time=float(observation["time_min"]), episode_id=episode_id,
                info={**info, "env_reward": float(env_reward), "rlaif_reward": learned_reward,
                      "reward_decomposition": reward_decomposition(info)},
            ))
        observation = next_observation
    return _episode_summary(env, env_total, rlaif_total, bus_charging_count)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def update_mappo(
    assignment_actor: AssignmentActor, bus_actor: BusActor, critic: CentralizedCritic,
    assignment_optimizer, bus_optimizer, critic_optimizer, buffer: AsyncMAPPOBuffer,
    training: dict[str, Any], rng: np.random.Generator,
) -> dict[str, float]:
    buffer.compute_returns_and_advantages(float(training["gamma"]), float(training["gae_lambda"]))
    aggregates = {key: [] for key in (
        "assignment_policy_loss", "bus_policy_loss", "value_loss", "entropy_assignment", "entropy_bus",
        "approx_kl_assignment", "approx_kl_bus", "clip_fraction_assignment", "clip_fraction_bus",
    )}
    indices_by_agent = {
        agent: np.asarray([i for i, item in enumerate(buffer.transitions) if item.agent_id == agent], dtype=int)
        for agent in ("assignment", "bus")
    }
    for _ in range(int(training["ppo_epochs"])):
        for agent_id, actor, optimizer in (
            ("assignment", assignment_actor, assignment_optimizer), ("bus", bus_actor, bus_optimizer)
        ):
            indices = indices_by_agent[agent_id]
            if not len(indices):
                continue
            indices = rng.permutation(indices)
            for start in range(0, len(indices), int(training["batch_size"])):
                batch = indices[start:start + int(training["batch_size"])]
                items = [buffer.transitions[i] for i in batch]
                observations = torch.tensor(np.asarray([item.local_obs for item in items]), dtype=torch.float32)
                actions = torch.tensor([item.action for item in items], dtype=torch.long)
                masks = torch.tensor(np.asarray([item.action_mask for item in items]), dtype=torch.bool)
                old_log_probs = torch.tensor([item.log_prob for item in items], dtype=torch.float32)
                advantages = torch.tensor(buffer.advantages[batch], dtype=torch.float32)
                new_log_probs, entropy = actor.evaluate_actions(observations, actions, masks)
                log_ratio = new_log_probs - old_log_probs
                ratio = log_ratio.exp()
                clipped = torch.clamp(ratio, 1.0 - float(training["clip_eps"]), 1.0 + float(training["clip_eps"]))
                policy_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
                loss = policy_loss - float(training["ent_coef"]) * entropy.mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), float(training["max_grad_norm"]))
                optimizer.step()
                prefix = "assignment" if agent_id == "assignment" else "bus"
                aggregates[f"{prefix}_policy_loss"].append(float(policy_loss.detach()))
                aggregates[f"entropy_{prefix}"].append(float(entropy.mean().detach()))
                aggregates[f"approx_kl_{prefix}"].append(float(((ratio - 1.0) - log_ratio).mean().detach()))
                aggregates[f"clip_fraction_{prefix}"].append(float(((ratio - 1.0).abs() > float(training["clip_eps"])).float().mean().detach()))
        for batch in buffer.minibatch_indices(int(training["batch_size"]), rng):
            states = torch.tensor(np.asarray([buffer.transitions[i].global_state for i in batch]), dtype=torch.float32)
            returns = torch.tensor(buffer.returns[batch], dtype=torch.float32)
            value_loss = nn.functional.mse_loss(critic(states), returns)
            critic_optimizer.zero_grad()
            (float(training["vf_coef"]) * value_loss).backward()
            nn.utils.clip_grad_norm_(critic.parameters(), float(training["max_grad_norm"]))
            critic_optimizer.step()
            aggregates["value_loss"].append(float(value_loss.detach()))
    result = {key: _mean(values) for key, values in aggregates.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("MAPPO update produced non-finite metrics")
    return result


def save_checkpoint(path, assignment_actor, bus_actor, critic, assignment_optimizer, bus_optimizer,
                    critic_optimizer, config, metrics) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "stage": 7, "algorithm": "asynchronous_mappo",
        "assignment_actor_state_dict": assignment_actor.state_dict(),
        "bus_actor_state_dict": bus_actor.state_dict(), "critic_state_dict": critic.state_dict(),
        "assignment_optimizer_state_dict": assignment_optimizer.state_dict(),
        "bus_optimizer_state_dict": bus_optimizer.state_dict(), "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "config": config, "action_mappings": {"assignment": "0=TD,1..H=TBD,H+1..2H=TLD", "bus": list(BUS_CHARGING_ACTIONS)},
        "feature_schema_version": FEATURE_SCHEMA_VERSION, "training_metrics": metrics,
        "dimensions": {"assignment_obs": assignment_actor.obs_dim, "bus_obs": bus_actor.obs_dim,
                       "stations": assignment_actor.station_count, "global_state": critic.global_state_dim},
    }, path)


def load_checkpoint(path):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != 7 or checkpoint.get("algorithm") != "asynchronous_mappo":
        raise ValueError("Checkpoint is not a Stage 7 asynchronous MAPPO checkpoint")
    dims, config = checkpoint["dimensions"], checkpoint["config"]
    assignment = AssignmentActor(dims["assignment_obs"], dims["stations"], config["networks"]["assignment_hidden_dims"])
    bus = BusActor(dims["bus_obs"], config["networks"]["bus_hidden_dims"])
    critic = CentralizedCritic(dims["global_state"], config["networks"]["critic_hidden_dims"])
    assignment.load_state_dict(checkpoint["assignment_actor_state_dict"])
    bus.load_state_dict(checkpoint["bus_actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    assignment.eval(); bus.eval(); critic.eval()
    return assignment, bus, critic, checkpoint


def _models(env, config):
    if tuple(env.config["bus"]["charging_actions_sec"]) != BUS_CHARGING_ACTIONS:
        raise ValueError(f"Stage 7 bus actions must be {list(BUS_CHARGING_ACTIONS)}")
    observation, _ = env.reset(seed=int(config["training"]["seed"]))
    assignment_dim = len(observation["features"])
    while observation["agent_id"] not in {"bus", "terminal"}:
        observation, *_ = env.step(next(i for i, allowed in enumerate(observation["action_mask"]) if allowed))
    bus_dim = len(observation["features"]) if observation["agent_id"] == "bus" else 6
    return (
        AssignmentActor(assignment_dim, len(env.station_ids), config["networks"]["assignment_hidden_dims"]),
        BusActor(bus_dim, config["networks"]["bus_hidden_dims"]),
        CentralizedCritic(len(env.get_global_state()), config["networks"]["critic_hidden_dims"]),
    )


def train_mappo_async(config: dict[str, Any], *, output_root=None) -> dict[str, Any]:
    training, seed = config["training"], int(config["training"]["seed"])
    set_seed(seed)
    env = create_environment(config, output_root=output_root)
    assignment, bus, critic = _models(env, config)
    assignment_optimizer = torch.optim.Adam(assignment.parameters(), lr=float(training["lr_actor"]))
    bus_optimizer = torch.optim.Adam(bus.parameters(), lr=float(training["lr_actor"]))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(training["lr_critic"]))
    wrapper = RewardModelWrapper(config["rlaif"].get("reward_model_checkpoint"), enabled=bool(config["rlaif"].get("enabled", False)))
    buffer, rng, rows = AsyncMAPPOBuffer(), np.random.default_rng(seed), []
    rollout_episodes = int(training["rollout_episodes"])
    for start in range(0, int(training["total_episodes"]), rollout_episodes):
        summaries = [collect_episode(env, assignment, bus, critic, buffer, wrapper, episode_id=seed + episode,
                     lambda_rlaif=float(config["rlaif"].get("lambda_rlaif", 1.0)))
                     for episode in range(start, min(start + rollout_episodes, int(training["total_episodes"])))]
        update = update_mappo(assignment, bus, critic, assignment_optimizer, bus_optimizer, critic_optimizer, buffer, training, rng)
        rows.extend({"episode": start + offset + 1, **summary, **update} for offset, summary in enumerate(summaries))
        buffer.clear()
    path = Path(config["output"]["training_log_path"]); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", *METRIC_FIELDS]); writer.writeheader(); writer.writerows(rows)
    save_checkpoint(config["output"]["checkpoint_path"], assignment, bus, critic, assignment_optimizer, bus_optimizer,
                    critic_optimizer, config, rows)
    return {"rows": rows, "checkpoint_path": config["output"]["checkpoint_path"], "models": (assignment, bus, critic)}


def evaluate_mappo_async(config: dict[str, Any], checkpoint_path, *, output_root=None, episodes: int = 1):
    env = create_environment(config, output_root=output_root)
    assignment, bus, critic, _ = load_checkpoint(checkpoint_path)
    wrapper = RewardModelWrapper(config["rlaif"].get("reward_model_checkpoint"), enabled=bool(config["rlaif"].get("enabled", False)))
    results = [collect_episode(env, assignment, bus, critic, None, wrapper,
               episode_id=int(config["training"]["seed"]) + index,
               lambda_rlaif=float(config["rlaif"].get("lambda_rlaif", 1.0)), deterministic=True)
               for index in range(episodes)]
    summary = {key: float(np.mean([float(row[key]) for row in results])) for key in results[0]}
    payload = {"checkpoint": str(checkpoint_path), "episodes": results, "summary": summary}
    path = Path(config["output"]["eval_path"]); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
