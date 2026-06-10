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
    assert len(observation["features"]) == 6
    assert len(observation["action_mask"]) == environment.assignment_action_size
    assert environment.assignment_action_size == 1 + 2 * len(environment.station_ids)
    assert observation["action_mask"][0]
    assert info["total_parcels"] == 60
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
