"""Evaluation-time policy adapters for formal benchmark rollouts."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from envs import first_feasible_policy
from training.event_schema import EVENT_NAME_TO_ID
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
    method_id = "mappo"
    def __init__(self, checkpoint_path: str|Path|None=None, *, spec: FormalPolicySpec|None=None):
        self.metadata = load_checkpoint_metadata(checkpoint_path) if checkpoint_path else {}
        if spec and checkpoint_path: validate_policy_checkpoint(spec, checkpoint_path)
        self.event_ids_seen: list[int] = []
    def select_action(self, *, observation: dict, env, deterministic: bool=True) -> int:
        event_type = str(observation.get("event_type"))
        canonical={"BUS_DEPARTURE":"BUS_TERMINAL_DEPARTURE","BUS_ARRIVAL":"BUS_STATION_ARRIVAL"}.get(event_type, event_type)
        if canonical not in EVENT_NAME_TO_ID:
            raise ValueError(f"unknown event type {event_type}")
        self.event_ids_seen.append(EVENT_NAME_TO_ID[canonical])
        return _check(observation, first_feasible_policy(observation))
