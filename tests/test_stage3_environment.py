"""Stage 3 event-driven MDP regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_pipeline.build_instance import build_instance
from envs import DynamicDeliveryEnv, InstanceValidationError, first_feasible_policy
from experiments.smoke_test_environment import run_smoke_test

CONFIG = Path(__file__).parents[1] / "configs/shanghai_small.yaml"


@pytest.fixture()
def environment(tmp_path: Path) -> DynamicDeliveryEnv:
    instance = build_instance(CONFIG, fallback=True, output_root=tmp_path)
    return DynamicDeliveryEnv(Path(instance["output_directory"]) / "instance.json")


def advance_to_agent(env: DynamicDeliveryEnv, agent: str) -> dict[str, object]:
    observation, _ = env.reset()
    while observation["agent"] not in {agent, "terminal"}:
        observation, *_ = env.step(first_feasible_policy(observation))
    return observation


def test_reset_exposes_stable_assignment_schema(environment: DynamicDeliveryEnv) -> None:
    observation, info = environment.reset(seed=7)

    assert observation["agent"] == "assignment"
    assert observation["agent_id"] == "assignment"
    assert observation["event_type"] == "PARCEL_ARRIVAL"
    assert len(observation["features"]) == 6
    assert len(environment.get_global_state()) == 15
    assert len(observation["action_mask"]) == environment.assignment_action_size
    assert environment.assignment_action_size == 1 + 2 * len(environment.station_ids)
    assert observation["action_mask"][0]
    assert info["total_parcels"] == 60
    assert "reward_components" in info
    assert set(info["metrics"]) >= {"decision_events", "assignment_events", "bus_charging_events", "delivered_parcels", "undelivered_parcels", "drone_deliveries", "total_reward", "infeasible_action_corrections"}
    assert environment.config["bus"]["charging_actions_sec"] == [0, 15, 30, 45, 60, 75, 90, 105, 120]
    assert environment.check_invariants() == []


def test_infeasible_assignment_is_corrected_and_penalized(environment: DynamicDeliveryEnv) -> None:
    observation, _ = environment.reset()
    while all(observation["action_mask"]):
        observation, *_ = environment.step(0)
    invalid_action = observation["action_mask"].index(False)

    _observation, reward, _terminated, _truncated, info = environment.step(invalid_action)

    assert info["action_corrected"] is True
    assert info["applied_action"] != invalid_action
    assert reward <= -float(environment.config["reward"]["infeasible_action"])
    assert environment.cost_components["infeasible_action"] > 0
    assert environment.check_invariants() == []


def test_bus_decision_uses_configured_charging_actions(environment: DynamicDeliveryEnv) -> None:
    observation = advance_to_agent(environment, "bus")

    assert observation["agent"] == "bus"
    assert len(observation["features"]) == 6
    assert len(observation["action_mask"]) == environment.bus_action_size
    assert observation["action_mask"][0]
    trip_id = str(observation["entity_id"]).split(":", 1)[0]
    soc_before = environment.bus_soc_kwh[trip_id]
    charge_action = next((i for i, feasible in enumerate(observation["action_mask"]) if i > 0 and feasible), 0)
    _next, reward, *_ = environment.step(charge_action)

    assert environment.bus_soc_kwh[trip_id] >= soc_before
    assert reward <= 0
    assert environment.check_invariants() == []


def test_stage3_offline_smoke_completes_episode(tmp_path: Path) -> None:
    result = run_smoke_test(CONFIG, output_root=tmp_path)

    assert result["decisions"]["assignment"] == result["total_parcels"]
    assert 0 < result["delivered_parcels"] <= result["total_parcels"]
    assert result["decisions"]["bus"] > 0
    assert result["invariants"] == "passed"


def test_rejects_non_stage2_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "instance.json"
    manifest.write_text(json.dumps({"stage": 1}), encoding="utf-8")

    with pytest.raises(InstanceValidationError, match="stage: 2"):
        DynamicDeliveryEnv(manifest)


def test_station_power_overload_is_soft_penalty(environment: DynamicDeliveryEnv) -> None:
    observation = advance_to_agent(environment, "bus")
    station_id = str(observation["entity_id"]).split(":", 1)[1]
    station = environment.stations[station_id]
    station.power_capacity_kw = 1.0
    mask = environment._bus_mask(environment.current_decision.event)

    assert mask[1] is True
    _next, reward, *_rest, info = environment.step(1)
    assert reward < 0
    assert info["cost_components"]["power_overload"] > 0


def test_station_drone_cycle_preserves_non_negative_resources(environment: DynamicDeliveryEnv) -> None:
    observation, _ = environment.reset()
    station_action = 1 + len(environment.station_ids)
    while observation["agent"] != "terminal":
        if observation["agent"] == "assignment" and observation["action_mask"][station_action]:
            action = station_action
        else:
            action = first_feasible_policy(observation)
        observation, *_ = environment.step(action)

    assert any(parcel.mode == "TLD" for parcel in environment.parcels.values())
    assert all(station.locker_load_kg >= 0 for station in environment.stations.values())
    assert all(station.full_batteries >= 0 for station in environment.stations.values())
    assert environment.check_invariants() == []
