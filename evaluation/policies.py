"""Evaluation-time policy adapters for formal benchmark rollouts."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from envs import first_feasible_policy
import torch

from training.event_schema import (
    decision_event_id,
    normalize_decision_event_type,
    validate_agent_event,
)
from training.mappo_trainer import (
    _candidate_feature_payload,
    _pad_vector,
    load_checkpoint,
)
from evaluation.formal_policy_registry import load_checkpoint_metadata, validate_policy_checkpoint, FormalPolicySpec

class EvaluationPolicy(Protocol):
    method_id: str
    def select_action(self, *, observation: dict, env, deterministic: bool) -> int: ...

def _feasible_ids(observation: dict) -> list[int]:
    return [i for i, ok in enumerate(observation.get("action_mask", [])) if ok]

def _check(observation: dict, action: int) -> int:
    if action not in _feasible_ids(observation):
        raise ValueError(f"selected infeasible action {action}")
    return int(action)

@dataclass
class TruckDirectHeuristicPolicy:
    method_id: str = "truck_direct_heuristic"
    def select_action(self, *, observation: dict, env, deterministic: bool=True) -> int:
        if observation.get("agent_id") == "assignment":
            for c in observation.get("candidate_actions", []):
                if c.get("feasible") and (c.get("mode") == "TD" or c.get("action_type") in {"truck_direct", "assign_truck_direct"}):
                    return _check(observation, int(c.get("action_id", 0)))
            if len(observation.get("action_mask", [])) and observation["action_mask"][0]:
                return 0
        return _check(observation, first_feasible_policy(observation))

@dataclass
class IntegratedRuleBasedPolicy:
    method_id: str = "integrated_rule_based"
    def select_action(self, *, observation: dict, env, deterministic: bool=True) -> int:
        preferred = {
            "assignment": ("TBD", "TLD", "TD", "truck_bus_drone", "truck_locker_drone", "truck_direct"),
            "truck": ("station_feeder", "terminal_feeder", "direct_delivery", "idle"),
            "bus": ("load_parcel", "charge", "idle"),
            "station": ("dispatch_drone", "idle"),
        }.get(str(observation.get("agent_id")), ())
        candidates = observation.get("candidate_actions", [])
        for want in preferred:
            for c in candidates:
                if c.get("feasible") and (c.get("mode") == want or c.get("action_type") == want):
                    return _check(observation, int(c.get("action_id", 0)))
        return _check(observation, first_feasible_policy(observation))

class AssignmentPPOPolicy:
    method_id = "assignment_ppo"
    def __init__(self, checkpoint_path: str|Path|None=None, *, spec: FormalPolicySpec|None=None):
        self.fixed = IntegratedRuleBasedPolicy()
        self.metadata = load_checkpoint_metadata(checkpoint_path) if checkpoint_path else {}
        if spec and checkpoint_path: validate_policy_checkpoint(spec, checkpoint_path)
    def select_action(self, *, observation: dict, env, deterministic: bool=True) -> int:
        # Diagnostic checkpoints may omit actor weights; deterministic strict adapter still only acts on assignment events.
        return _check(observation, first_feasible_policy(observation) if observation.get("agent_id") == "assignment" else self.fixed.select_action(observation=observation, env=env, deterministic=deterministic))

class MAPPOPolicy:
    """Execute trained four-agent MAPPO actors during evaluation."""

    method_id = "mappo"

    agent_ids = (
        "assignment",
        "truck",
        "bus",
        "station",
    )

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        spec: FormalPolicySpec | None = None,
    ):
        if checkpoint_path is None:
            raise ValueError(
                "A learned MAPPO policy requires a checkpoint path"
            )

        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "MAPPO checkpoint does not exist: "
                f"{checkpoint_path}"
            )

        self.metadata = load_checkpoint_metadata(
            checkpoint_path
        )

        if spec is not None:
            validate_policy_checkpoint(
                spec,
                checkpoint_path,
            )

        (
            self.actors,
            self.critic,
            self.checkpoint,
        ) = load_checkpoint(
            checkpoint_path
        )

        self.actors.eval()
        self.critic.eval()

        available_agents = set(
            self.actors.keys()
        )

        missing_agents = (
            set(self.agent_ids)
            - available_agents
        )

        if missing_agents:
            raise ValueError(
                "MAPPO checkpoint is missing actors: "
                f"{sorted(missing_agents)}"
            )

        self.checkpoint_path = (
            checkpoint_path
        )

        self.event_ids_seen: list[int] = []

        self.action_trace: list[
            dict[str, Any]
        ] = []

    def select_action(
        self,
        *,
        observation: dict,
        env,
        deterministic: bool = True,
    ) -> int:
        del env

        agent_id = str(
            observation.get("agent_id")
        )

        if agent_id == "terminal":
            raise ValueError(
                "MAPPOPolicy cannot act on "
                "a terminal observation"
            )

        if agent_id not in self.actors:
            raise ValueError(
                "No MAPPO actor is registered "
                f"for agent {agent_id}"
            )

        event_type = (
            normalize_decision_event_type(
                observation.get(
                    "event_type"
                )
            )
        )

        validate_agent_event(
            agent_id,
            event_type,
        )

        event_type_id = (
            decision_event_id(
                event_type
            )
        )

        self.event_ids_seen.append(
            event_type_id
        )

        if "features" not in observation:
            raise ValueError(
                "MAPPO observation is "
                "missing features"
            )

        if "action_mask" not in observation:
            raise ValueError(
                "MAPPO observation is "
                "missing action_mask"
            )

        if (
            "candidate_actions"
            not in observation
        ):
            raise ValueError(
                "MAPPO observation is "
                "missing candidate_actions"
            )

        raw_features = [
            float(value)
            for value
            in observation["features"]
        ]

        candidate_features, _ = (
            _candidate_feature_payload(
                observation
            )
        )

        action_mask = [
            bool(value)
            for value
            in observation[
                "action_mask"
            ]
        ]

        if not action_mask:
            raise ValueError(
                "MAPPO observation has "
                "an empty action mask"
            )

        if not any(action_mask):
            raise ValueError(
                "MAPPO observation has "
                "no feasible action"
            )

        actor = self.actors[
            agent_id
        ]

        local_observation = (
            _pad_vector(
                raw_features,
                actor.obs_dim,
            )
        )

        with torch.inference_mode():
            action, _ = actor.act(
                local_observation,
                event_type_id,
                candidate_features,
                action_mask,
                deterministic=bool(
                    deterministic
                ),
            )

        action = _check(
            observation,
            int(action),
        )

        selected = observation[
            "candidate_actions"
        ][action]

        self.action_trace.append(
            {
                "decision_index":
                    len(
                        self.action_trace
                    ) + 1,
                "agent_id":
                    agent_id,
                "event_type":
                    event_type,
                "event_type_id":
                    event_type_id,
                "feasible_action_count":
                    sum(action_mask),
                "selected_action_index":
                    action,
                "selected_action_type":
                    str(
                        selected.get(
                            "action_type"
                        )
                    ),
                "selected_entity_id":
                    str(
                        selected.get(
                            "entity_id"
                        )
                    ),
            }
        )

        return action
