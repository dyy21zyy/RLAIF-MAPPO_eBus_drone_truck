"""Four-agent event-driven environment tests for the Solution Method pass."""

from __future__ import annotations

from pathlib import Path

from data_pipeline.build_instance import build_instance
from envs import DynamicDeliveryEnv, first_feasible_policy

CONFIG = Path(__file__).parents[1] / "configs/shanghai_small.yaml"


def make_env(tmp_path: Path) -> DynamicDeliveryEnv:
    instance = build_instance(CONFIG, fallback=True, output_root=tmp_path)
    return DynamicDeliveryEnv(Path(instance["output_directory"]) / "instance.json")


def four_agent_exercising_policy(env: DynamicDeliveryEnv, observation: dict[str, object]) -> int:
    mask = [bool(value) for value in observation["action_mask"]]
    if observation["agent_id"] == "assignment":
        first_tld = 1 + len(env.station_ids)
        for action_id in range(first_tld, len(mask)):
            if mask[action_id]:
                return action_id
        for action_id in range(1, first_tld):
            if mask[action_id]:
                return action_id
    if observation["agent_id"] == "truck":
        for candidate in observation["candidate_actions"]:
            if candidate["feasible"] and candidate["action_type"] == "station_feeder":
                return int(candidate["action_id"])
    if observation["agent_id"] == "station":
        for candidate in observation["candidate_actions"]:
            if candidate["feasible"] and candidate["action_type"] == "dispatch_drone":
                return int(candidate["action_id"])
    return first_feasible_policy(observation)


def collect_agents(env: DynamicDeliveryEnv, limit: int = 500) -> list[tuple[str, str]]:
    observation, _ = env.reset(seed=11)
    seen: list[tuple[str, str]] = []
    while observation["agent_id"] != "terminal" and len(seen) < limit:
        seen.append((str(observation["agent_id"]), str(observation["event_type"])))
        assert "candidate_actions" in observation
        assert "candidate_features" in observation
        assert "candidate_feature_names" in observation
        assert len(observation["candidate_actions"]) == len(observation["action_mask"])
        assert len(observation["candidate_features"]) == len(observation["action_mask"])
        assert any(observation["action_mask"])
        observation, *_ = env.step(four_agent_exercising_policy(env, observation))
        assert env.check_invariants() == []
    return seen


def test_episode_exposes_assignment_truck_bus_and_station_decisions(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    seen = collect_agents(env)
    agent_ids = {agent for agent, _event in seen}
    assert {"assignment", "truck", "bus", "station"} <= agent_ids


def test_four_agent_event_types_are_operational_not_dummy(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    seen = collect_agents(env)
    pairs = set(seen)
    assert ("assignment", "PARCEL_RELEASE") in pairs
    assert ("truck", "TRUCK_AVAILABLE") in pairs
    assert ("bus", "BUS_DEPARTURE") in pairs
    assert ("bus", "BUS_ARRIVAL") in pairs
    assert ("station", "STATION_OPERATION") in pairs
    assert all(agent != "inactive" for agent, _event in seen)
