"""Shared four-agent candidate-action schema tests."""

from __future__ import annotations

import pytest

from envs.decision_schema import ActionCandidate, DecisionSurface


def test_decision_surface_exports_mask_and_candidate_matrix() -> None:
    surface = DecisionSurface(
        agent_id="truck",
        event_type="TRUCK_AVAILABLE",
        entity_id="truck_000",
        features=[0.25, 1.0],
        feature_names=("time_norm", "capacity_norm"),
        candidates=[
            ActionCandidate(
                action_id=0,
                action_type="execute_task",
                entity_id="parcel_001",
                description="direct delivery",
                features={"estimated_time_norm": 0.2, "idle_flag": 0.0},
                feasible=True,
                reasons=(),
            ),
            ActionCandidate(
                action_id=1,
                action_type="idle",
                entity_id="truck_000",
                description="remain idle",
                features={"estimated_time_norm": 0.0, "idle_flag": 1.0},
                feasible=False,
                reasons=("task_available",),
            ),
        ],
    )

    assert surface.action_mask() == [True, False]
    assert surface.candidate_feature_names() == ("estimated_time_norm", "idle_flag")
    assert surface.candidate_feature_matrix() == [[0.2, 0.0], [0.0, 1.0]]
    payload = surface.candidate_payloads()[0]
    assert payload["action_type"] == "execute_task"
    assert payload["features"]["estimated_time_norm"] == 0.2


def test_decision_surface_requires_one_feasible_candidate() -> None:
    with pytest.raises(ValueError, match="at least one feasible"):
        DecisionSurface(
            agent_id="station",
            event_type="STATION_OPERATION",
            entity_id="station_001",
            features=[0.0],
            feature_names=("time_norm",),
            candidates=[
                ActionCandidate(
                    0,
                    "dispatch",
                    "parcel_001",
                    "dispatch",
                    {"x": 1.0},
                    False,
                    ("no_battery",),
                )
            ],
        )
