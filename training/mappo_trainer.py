"""Rollout collection, PPO updates, checkpoints, and evaluation for Stage 9."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from training.mappo_async import (
    CANDIDATE_SCHEMA_VERSION,
    EVENT_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    RLAIF_AGENT_TYPES,
    VALID_AGENT_EVENTS,
    reward_decomposition,
    transition_reward,
    validate_decision,
)
from training.mappo_buffer import AsyncMAPPOBuffer, AsyncTransition
from training.entity_encoders import ENTITY_ENCODER_SCHEMA_VERSION
from training.mappo_networks import EVENT_EMBEDDING_SCHEMA_VERSION, CandidateScoringActor, CentralizedCritic, build_actor_registry
from training.ppo_trainer import create_environment
from training.reward_model_wrapper import RewardModelWrapper
from rlaif.reward_registry import RewardRegistry

AGENT_IDS = ("assignment", "truck", "bus", "station")
FEATURE_SCHEMA_VERSION = 3
CHECKPOINT_SCHEMA_VERSION = 3
ACTOR_POLICY_FIELDS = tuple(f"{agent}_policy_loss" for agent in AGENT_IDS)
ACTOR_ENTROPY_FIELDS = tuple(f"entropy_{agent}" for agent in AGENT_IDS)
ACTOR_KL_FIELDS = tuple(f"approx_kl_{agent}" for agent in AGENT_IDS)
ACTOR_CLIP_FIELDS = tuple(f"clip_fraction_{agent}" for agent in AGENT_IDS)
METRIC_FIELDS = (
    "episode_reward",
    "episode_env_reward",
    "episode_rlaif_reward",
    "assignment_decision_count",
    "truck_decision_count",
    "bus_decision_count",
    "station_decision_count",
    *ACTOR_POLICY_FIELDS,
    "value_loss",
    *ACTOR_ENTROPY_FIELDS,
    *ACTOR_KL_FIELDS,
    *ACTOR_CLIP_FIELDS,
    "delivered_parcels",
    "undelivered_parcels",
    "average_lateness",
    "infeasible_action_count",
    "bus_charging_count",
    "passenger_delay",
    "bus_operating_delay",
    "power_overload_amount",
    "locker_overflow_amount",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _candidate_payload(observation: dict[str, Any], action: int) -> dict[str, Any]:
    return dict(observation["candidate_actions"][action])


def _candidate_feature_payload(observation: dict[str, Any]) -> tuple[list[list[float]], tuple[str, ...]]:
    names = tuple(str(name) for name in observation["candidate_feature_names"])
    features = [
        [float(value) for value in row]
        for row in observation["candidate_features"]
    ]
    if len(features) != len(observation["action_mask"]):
        raise ValueError("candidate_features and action_mask differ in length")
    if any(len(row) != len(names) for row in features):
        raise ValueError("candidate feature rows do not match candidate_feature_names")
    return features, names


def _pad_vector(values: Sequence[float], target_dim: int) -> list[float]:
    result = [float(value) for value in values]
    if len(result) > target_dim:
        raise ValueError("Observation vector exceeds registered actor dimension")
    return result + [0.0] * (target_dim - len(result))


def _episode_summary(
    env,
    env_reward: float,
    rlaif_reward: float,
    bus_charging_count: int,
) -> dict[str, float | int]:
    delivered = [parcel for parcel in env.parcels.values() if parcel.status == "delivered"]
    lateness = [max(0.0, float(parcel.delivered_time_min) - parcel.deadline_min) for parcel in delivered]
    costs = env.cost_components
    return {
        "episode_reward": env_reward + rlaif_reward,
        "episode_env_reward": env_reward,
        "episode_rlaif_reward": rlaif_reward,
        "assignment_decision_count": env.decision_counts["assignment"],
        "truck_decision_count": env.decision_counts["truck"],
        "bus_decision_count": env.decision_counts["bus"],
        "station_decision_count": env.decision_counts["station"],
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
    env,
    actors: nn.ModuleDict,
    critic: CentralizedCritic,
    buffer: AsyncMAPPOBuffer | None,
    reward_wrapper: RewardModelWrapper | RewardRegistry,
    *,
    episode_id: int,
    lambda_rlaif: float,
    deterministic: bool = False,
) -> dict[str, float | int]:
    """Act once and append once for each actual four-agent decision event."""

    observation, _ = env.reset(seed=episode_id)
    env_total = 0.0
    rlaif_total = 0.0
    bus_charging_count = 0
    while observation["agent_id"] != "terminal":
        agent_id = str(observation["agent_id"])
        event_type = str(observation["event_type"])
        validate_decision(agent_id, event_type)
        if agent_id not in actors:
            raise RuntimeError(f"No MAPPO actor registered for {agent_id}")
        mask = [bool(value) for value in observation["action_mask"]]
        candidate_features, candidate_feature_names = _candidate_feature_payload(observation)
        global_state = [float(value) for value in env.get_entity_critic_state()]
        actor = actors[agent_id]
        local_obs = _pad_vector(observation["features"], actor.obs_dim)
        action, log_prob = actor.act(
            local_obs, candidate_features, mask, deterministic=deterministic
        )
        if not mask[action]:
            raise RuntimeError("Masked MAPPO sampled an infeasible action")
        action_payload = _candidate_payload(observation, action)
        if agent_id == "bus" and action_payload.get("action_type") == "charge":
            bus_charging_count += 1
        with torch.no_grad():
            value = float(critic(torch.tensor(global_state, dtype=torch.float32)).item())

        action_features = candidate_features[action] if agent_id in RLAIF_AGENT_TYPES else None
        next_observation, env_reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        if isinstance(reward_wrapper, RewardRegistry):
            total_reward, learned_reward = reward_wrapper.total_reward(agent_id, event_type, float(env_reward), local_obs, action_features, action)
        else:
            total_reward, learned_reward = transition_reward(
                agent_id,
                float(env_reward),
                reward_wrapper,
                lambda_rlaif=lambda_rlaif,
                state_features=local_obs,
                action_features=action_features,
                action_id=action,
                event_type=event_type,
            )
        env_total += float(env_reward)
        rlaif_total += learned_reward * float(lambda_rlaif)
        if buffer is not None:
            buffer.append(
                AsyncTransition(
                    agent_id=agent_id,
                    local_obs=local_obs,
                    global_state=global_state,
                    action=action,
                    action_mask=mask,
                    candidate_features=candidate_features,
                    candidate_feature_names=candidate_feature_names,
                    log_prob=log_prob,
                    value=value,
                    reward=total_reward,
                    done=done,
                    next_global_state=[float(value) for value in env.get_entity_critic_state()],
                    event_type=event_type,
                    event_time=float(observation["time_min"]),
                    episode_id=episode_id,
                    info={
                        **info,
                        "env_reward": float(env_reward),
                        "rlaif_reward": learned_reward,
                        "action_candidate": action_payload,
                        "reward_decomposition": reward_decomposition(info),
                    },
                )
            )
        observation = next_observation
    return _episode_summary(env, env_total, rlaif_total, bus_charging_count)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _padded_candidate_batch(items: Sequence[AsyncTransition]) -> tuple[torch.Tensor, torch.Tensor]:
    if not items:
        raise ValueError("Cannot build an empty candidate batch")
    feature_names = items[0].candidate_feature_names
    feature_dim = len(feature_names)
    if any(item.candidate_feature_names != feature_names for item in items):
        raise ValueError("Candidate feature schema differs within an agent batch")
    max_actions = max(len(item.action_mask) for item in items)
    candidates = np.zeros((len(items), max_actions, feature_dim), dtype=np.float32)
    masks = np.zeros((len(items), max_actions), dtype=bool)
    for row_index, item in enumerate(items):
        rows = np.asarray(item.candidate_features, dtype=np.float32)
        candidates[row_index, : rows.shape[0], :] = rows
        masks[row_index, : len(item.action_mask)] = np.asarray(item.action_mask, dtype=bool)
    return torch.tensor(candidates, dtype=torch.float32), torch.tensor(masks, dtype=torch.bool)


def update_mappo(
    actors: nn.ModuleDict,
    critic: CentralizedCritic,
    actor_optimizers: dict[str, torch.optim.Optimizer],
    critic_optimizer: torch.optim.Optimizer,
    buffer: AsyncMAPPOBuffer,
    training: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, float]:
    buffer.compute_returns_and_advantages(
        float(training["gamma"]),
        float(training["gae_lambda"]),
        reference_time_unit=float(training.get("event_time_reference_min", 1.0)),
    )
    aggregates = {key: [] for key in (
        *ACTOR_POLICY_FIELDS,
        "value_loss",
        *ACTOR_ENTROPY_FIELDS,
        *ACTOR_KL_FIELDS,
        *ACTOR_CLIP_FIELDS,
    )}
    indices_by_agent = {
        agent: np.asarray([i for i, item in enumerate(buffer.transitions) if item.agent_id == agent], dtype=int)
        for agent in AGENT_IDS
    }
    for _ in range(int(training["ppo_epochs"])):
        for agent_id, actor in actors.items():
            indices = indices_by_agent[str(agent_id)]
            if not len(indices):
                continue
            indices = rng.permutation(indices)
            optimizer = actor_optimizers[str(agent_id)]
            for start in range(0, len(indices), int(training["batch_size"])):
                batch = indices[start:start + int(training["batch_size"])]
                items = [buffer.transitions[i] for i in batch]
                observations = torch.tensor(
                    np.asarray([item.local_obs for item in items]), dtype=torch.float32
                )
                candidate_features, masks = _padded_candidate_batch(items)
                actions = torch.tensor([item.action for item in items], dtype=torch.long)
                old_log_probs = torch.tensor([item.log_prob for item in items], dtype=torch.float32)
                advantages = torch.tensor(buffer.advantages[batch], dtype=torch.float32)
                new_log_probs, entropy = actor.evaluate_actions(
                    observations, candidate_features, actions, masks
                )
                log_ratio = new_log_probs - old_log_probs
                ratio = log_ratio.exp()
                clipped = torch.clamp(
                    ratio, 1.0 - float(training["clip_eps"]), 1.0 + float(training["clip_eps"])
                )
                policy_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
                loss = policy_loss - float(training["ent_coef"]) * entropy.mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), float(training["max_grad_norm"]))
                optimizer.step()
                aggregates[f"{agent_id}_policy_loss"].append(float(policy_loss.detach()))
                aggregates[f"entropy_{agent_id}"].append(float(entropy.mean().detach()))
                aggregates[f"approx_kl_{agent_id}"].append(
                    float(((ratio - 1.0) - log_ratio).mean().detach())
                )
                aggregates[f"clip_fraction_{agent_id}"].append(
                    float(((ratio - 1.0).abs() > float(training["clip_eps"])).float().mean().detach())
                )
        for batch in buffer.minibatch_indices(int(training["batch_size"]), rng):
            states = torch.tensor(
                np.asarray([buffer.transitions[i].global_state for i in batch]), dtype=torch.float32
            )
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


def _actor_specs(actors: nn.ModuleDict) -> dict[str, dict[str, Any]]:
    specs = {}
    for agent_id, actor in actors.items():
        if not isinstance(actor, CandidateScoringActor):
            raise TypeError("Stage 9 checkpoint expects candidate-scoring actors")
        specs[str(agent_id)] = {
            "obs_dim": actor.obs_dim,
            "candidate_feature_dim": actor.candidate_feature_dim,
            "hidden_dims": list(actor.hidden_dims),
        }
    return specs


def save_checkpoint(
    path,
    actors: nn.ModuleDict,
    critic: CentralizedCritic,
    actor_optimizers: dict[str, torch.optim.Optimizer],
    critic_optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    metrics: list[dict[str, Any]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "stage": 7,
            "algorithm": "four_agent_asynchronous_mappo_env_reward_only",
            "actor_state_dicts": actors.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "actor_optimizer_state_dicts": {
                agent: optimizer.state_dict() for agent, optimizer in actor_optimizers.items()
            },
            "critic_optimizer_state_dict": critic_optimizer.state_dict(),
            "config": config,
            "action_mappings": {
                "assignment": "0=TD,1..H=TBD,H+1..2H=TLD",
                "truck": "candidate task index or idle",
                "bus": "BUS_DEPARTURE load/idle or BUS_ARRIVAL charge/no_charge",
                "station": "dispatch_drone or idle",
            },
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "event_embedding_schema_version": EVENT_EMBEDDING_SCHEMA_VERSION,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
            "entity_encoder_schema_version": ENTITY_ENCODER_SCHEMA_VERSION,
            "reward_scale_artifact_hash": config.get("reward", {}).get("scale_artifact_hash", "unavailable"),
            "training_seed": config.get("training", {}).get("seed"),
            "code_commit": _code_commit(),
            "training_metrics": metrics,
            "actor_specs": _actor_specs(actors),
            "dimensions": {
                "actors": _actor_specs(actors),
                "global_state": critic.global_state_dim,
            },
        },
        path,
    )


def _code_commit() -> str | None:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None

def _require_checkpoint_compatible(checkpoint: dict[str, Any]) -> None:
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "entity_encoder_schema_version": ENTITY_ENCODER_SCHEMA_VERSION,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Incompatible MAPPO checkpoint {key}: expected {value}, got {checkpoint.get(key)}")
    if checkpoint.get("stage") != 7 or checkpoint.get("algorithm") != "four_agent_asynchronous_mappo_env_reward_only":
        raise ValueError("Checkpoint is not a Phase 7 four-agent asynchronous MAPPO environment-reward checkpoint")

def load_checkpoint(path):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    _require_checkpoint_compatible(checkpoint)
    specs_payload = checkpoint["actor_specs"]
    specs = {
        agent: (int(payload["obs_dim"]), int(payload["candidate_feature_dim"]))
        for agent, payload in specs_payload.items()
    }
    hidden_dims = {
        agent: list(payload["hidden_dims"])
        for agent, payload in specs_payload.items()
    }
    hidden_dims["default"] = list(next(iter(specs_payload.values()))["hidden_dims"])
    actors = build_actor_registry(specs, hidden_dims)
    critic = CentralizedCritic(
        int(checkpoint["dimensions"]["global_state"]),
        checkpoint["config"]["networks"]["critic_hidden_dims"],
    )
    actors.load_state_dict(checkpoint["actor_state_dicts"])
    critic.load_state_dict(checkpoint["critic_state_dict"])
    actors.eval()
    critic.eval()
    return actors, critic, checkpoint


def _first_candidate_action(
    observation: dict[str, Any], action_type: str
) -> int | None:
    for candidate in observation["candidate_actions"]:
        if candidate["feasible"] and candidate["action_type"] == action_type:
            return int(candidate["action_id"])
    return None


def _dimension_probe_action(env, observation: dict[str, Any]) -> int:
    mask = [bool(value) for value in observation["action_mask"]]
    agent_id = str(observation["agent_id"])
    if agent_id == "assignment":
        first_tld = 1 + len(env.station_ids)
        for action_id in range(first_tld, len(mask)):
            if mask[action_id]:
                return action_id
        for action_id in range(1, first_tld):
            if mask[action_id]:
                return action_id
    if agent_id == "truck":
        action = _first_candidate_action(observation, "station_feeder")
        if action is not None:
            return action
    if agent_id == "station":
        action = _first_candidate_action(observation, "dispatch_drone")
        if action is not None:
            return action
    return next(index for index, feasible in enumerate(mask) if feasible)


def _collect_actor_specs(env, seed: int, max_steps: int = 2000) -> dict[str, tuple[int, int]]:
    observation, _ = env.reset(seed=seed)
    specs: dict[str, tuple[int, int]] = {}
    observed_pairs: set[tuple[str, str]] = set()
    steps = 0
    while observation["agent_id"] != "terminal" and steps < max_steps:
        agent_id = str(observation["agent_id"])
        event_type = str(observation["event_type"])
        candidate_features, candidate_names = _candidate_feature_payload(observation)
        del candidate_features
        observed_pairs.add((agent_id, event_type))
        obs_dim = len(observation["features"])
        candidate_dim = len(candidate_names)
        if agent_id in specs:
            previous_obs_dim, previous_candidate_dim = specs[agent_id]
            if previous_candidate_dim != candidate_dim:
                raise RuntimeError(f"Candidate feature dimension changed for {agent_id}")
            specs[agent_id] = (max(previous_obs_dim, obs_dim), candidate_dim)
        else:
            specs[agent_id] = (obs_dim, candidate_dim)
        if set(specs) == set(AGENT_IDS) and ("bus", "BUS_ARRIVAL") in observed_pairs:
            break
        observation, *_ = env.step(_dimension_probe_action(env, observation))
        errors = env.check_invariants()
        if errors:
            raise RuntimeError(f"Dimension probe violated environment invariants: {errors}")
        steps += 1
    missing = set(AGENT_IDS) - set(specs)
    if missing:
        raise RuntimeError(f"Could not observe Stage 9 agent dimensions for {sorted(missing)}")
    return specs


def _actor_hidden_dims(config: dict[str, Any]) -> dict[str, Sequence[int]]:
    networks = config["networks"]
    default = networks.get("actor_hidden_dims", networks.get("assignment_hidden_dims", [256, 256]))
    return {
        "default": default,
        "assignment": networks.get("assignment_hidden_dims", default),
        "truck": networks.get("truck_hidden_dims", default),
        "bus": networks.get("bus_hidden_dims", default),
        "station": networks.get("station_hidden_dims", default),
    }


def _models(env, config):
    specs = _collect_actor_specs(env, int(config["training"]["seed"]))
    return (
        build_actor_registry(specs, _actor_hidden_dims(config)),
        CentralizedCritic(len(env.get_entity_critic_state()), config["networks"]["critic_hidden_dims"]),
    )


def train_mappo_async(config: dict[str, Any], *, output_root=None) -> dict[str, Any]:
    training, seed = config["training"], int(config["training"]["seed"])
    set_seed(seed)
    env = create_environment(config, output_root=output_root)
    actors, critic = _models(env, config)
    actor_optimizers = {
        agent: torch.optim.Adam(actor.parameters(), lr=float(training["lr_actor"]))
        for agent, actor in actors.items()
    }
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(training["lr_critic"]))
    wrapper = (
        RewardRegistry(config)
        if "agents" in config.get("rlaif", {})
        else RewardModelWrapper(
            config["rlaif"].get("reward_model_checkpoint"),
            enabled=bool(config["rlaif"].get("enabled", False)),
            validation=config["rlaif"].get("validation", {}),
            fallback_to_env_reward=bool(config["rlaif"].get("fallback_to_env_reward", True)),
            fail_on_invalid_reward_model=bool(config["rlaif"].get("fail_on_invalid_reward_model", False)),
            reward_clip=config["rlaif"].get("reward_clip"),
        )
    )
    buffer, rng, rows = AsyncMAPPOBuffer(), np.random.default_rng(seed), []
    rollout_episodes = int(training["rollout_episodes"])
    for start in range(0, int(training["total_episodes"]), rollout_episodes):
        summaries = [
            collect_episode(
                env,
                actors,
                critic,
                buffer,
                wrapper,
                episode_id=seed + episode,
                lambda_rlaif=float(config.get("rlaif", {}).get("lambda", 0.0)),
            )
            for episode in range(start, min(start + rollout_episodes, int(training["total_episodes"])))
        ]
        update = update_mappo(
            actors,
            critic,
            actor_optimizers,
            critic_optimizer,
            buffer,
            training,
            rng,
        )
        rows.extend({"episode": start + offset + 1, **summary, **update} for offset, summary in enumerate(summaries))
        buffer.clear()
    path = Path(config["output"]["training_log_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", *METRIC_FIELDS])
        writer.writeheader()
        writer.writerows(rows)
    save_checkpoint(
        config["output"]["checkpoint_path"],
        actors,
        critic,
        actor_optimizers,
        critic_optimizer,
        config,
        rows,
    )
    return {"rows": rows, "checkpoint_path": config["output"]["checkpoint_path"], "models": (actors, critic)}


def evaluate_mappo_async(config: dict[str, Any], checkpoint_path, *, output_root=None, episodes: int = 1):
    env = create_environment(config, output_root=output_root)
    actors, critic, _ = load_checkpoint(checkpoint_path)
    wrapper = (
        RewardRegistry(config)
        if "agents" in config.get("rlaif", {})
        else RewardModelWrapper(
            config["rlaif"].get("reward_model_checkpoint"),
            enabled=bool(config["rlaif"].get("enabled", False)),
            validation=config["rlaif"].get("validation", {}),
            fallback_to_env_reward=bool(config["rlaif"].get("fallback_to_env_reward", True)),
            fail_on_invalid_reward_model=bool(config["rlaif"].get("fail_on_invalid_reward_model", False)),
            reward_clip=config["rlaif"].get("reward_clip"),
        )
    )
    results = [
        collect_episode(
            env,
            actors,
            critic,
            None,
            wrapper,
            episode_id=int(config["training"]["seed"]) + index,
            lambda_rlaif=float(config.get("rlaif", {}).get("lambda", 0.0)),
            deterministic=True,
        )
        for index in range(episodes)
    ]
    summary = {key: float(np.mean([float(row[key]) for row in results])) for key in results[0]}
    payload = {"checkpoint": str(checkpoint_path), "episodes": results, "summary": summary}
    path = Path(config["output"]["eval_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
