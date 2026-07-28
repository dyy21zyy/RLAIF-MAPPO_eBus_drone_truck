from __future__ import annotations

from pathlib import Path

import pytest

import evaluation.policies as policies


AGENT_IDS = (
    "assignment",
    "truck",
    "bus",
    "station",
)


class FakeActor:
    def __init__(
        self,
        *,
        selected_action: int,
        obs_dim: int = 2,
    ):
        self.selected_action = selected_action
        self.obs_dim = obs_dim
        self.calls: list[dict] = []

    def act(
        self,
        local_observation,
        event_type_id,
        candidate_features,
        action_mask,
        deterministic=True,
    ):
        self.calls.append(
            {
                "local_observation":
                    list(local_observation),
                "event_type_id":
                    int(event_type_id),
                "candidate_features":
                    candidate_features,
                "action_mask":
                    list(action_mask),
                "deterministic":
                    bool(deterministic),
            }
        )

        return self.selected_action, -0.5


class FakeActorRegistry(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self


class FakeCritic:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self


def make_observation(
    *,
    action_mask=(True, True),
):
    return {
        "agent_id": "assignment",
        "event_type": "PARCEL_RELEASE",
        "features": [
            0.25,
            -0.50,
        ],
        "candidate_feature_names": [
            "feature_a",
            "feature_b",
        ],
        "candidate_features": [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        "action_mask": list(
            action_mask
        ),
        "candidate_actions": [
            {
                "action_id": 0,
                "action_type": "TD",
                "entity_id": "parcel_0",
                "feasible":
                    bool(action_mask[0]),
            },
            {
                "action_id": 1,
                "action_type": "TBD",
                "entity_id": "parcel_0",
                "feasible":
                    bool(action_mask[1]),
            },
        ],
    }


def build_policy(
    monkeypatch,
    tmp_path: Path,
    *,
    selected_action: int,
    missing_agent: str | None = None,
):
    checkpoint_path = (
        tmp_path / "policy.pt"
    )

    checkpoint_path.write_bytes(
        b"fake-checkpoint"
    )

    actors = FakeActorRegistry(
        {
            agent: FakeActor(
                selected_action=(
                    selected_action
                    if agent == "assignment"
                    else 0
                )
            )
            for agent in AGENT_IDS
            if agent != missing_agent
        }
    )

    critic = FakeCritic()

    checkpoint = {
        "algorithm":
            "four_agent_asynchronous_mappo_env",
        "rlaif_scope": "none",
    }

    monkeypatch.setattr(
        policies,
        "load_checkpoint_metadata",
        lambda path: {
            "checkpoint_path":
                str(path),
        },
    )

    monkeypatch.setattr(
        policies,
        "load_checkpoint",
        lambda path: (
            actors,
            critic,
            checkpoint,
        ),
    )

    policy = policies.MAPPOPolicy(
        checkpoint_path,
        spec=None,
    )

    return policy, actors, critic


def test_mappo_policy_requires_checkpoint():
    with pytest.raises(
        ValueError,
        match="requires a checkpoint",
    ):
        policies.MAPPOPolicy(
            None,
            spec=None,
        )


def test_mappo_policy_loads_actors_and_sets_eval(
    monkeypatch,
    tmp_path,
):
    policy, actors, critic = (
        build_policy(
            monkeypatch,
            tmp_path,
            selected_action=1,
        )
    )

    assert set(
        policy.actors.keys()
    ) == set(AGENT_IDS)

    assert actors.eval_called
    assert critic.eval_called


def test_mappo_policy_uses_actor_not_first_feasible(
    monkeypatch,
    tmp_path,
):
    policy, actors, _ = (
        build_policy(
            monkeypatch,
            tmp_path,
            selected_action=1,
        )
    )

    observation = make_observation(
        action_mask=(
            True,
            True,
        )
    )

    # first_feasible_policy would choose
    # action 0. The fake actor chooses 1.
    action = policy.select_action(
        observation=observation,
        env=None,
        deterministic=True,
    )

    assert action == 1

    actor = actors["assignment"]

    assert len(actor.calls) == 1

    call = actor.calls[0]

    assert call["event_type_id"] == 0

    assert call["action_mask"] == [
        True,
        True,
    ]

    assert (
        call["deterministic"]
        is True
    )

    assert (
        policy.action_trace[-1][
            "selected_action_index"
        ]
        == 1
    )

    assert (
        policy.action_trace[-1][
            "selected_action_type"
        ]
        == "TBD"
    )


def test_mappo_policy_forwards_deterministic_flag(
    monkeypatch,
    tmp_path,
):
    policy, actors, _ = (
        build_policy(
            monkeypatch,
            tmp_path,
            selected_action=0,
        )
    )

    policy.select_action(
        observation=
            make_observation(),
        env=None,
        deterministic=False,
    )

    assert (
        actors["assignment"]
        .calls[0]["deterministic"]
        is False
    )


def test_mappo_policy_rejects_masked_actor_action(
    monkeypatch,
    tmp_path,
):
    policy, _, _ = build_policy(
        monkeypatch,
        tmp_path,
        selected_action=1,
    )

    observation = make_observation(
        action_mask=(
            True,
            False,
        )
    )

    with pytest.raises(
        ValueError,
        match="selected infeasible action",
    ):
        policy.select_action(
            observation=observation,
            env=None,
            deterministic=True,
        )


def test_mappo_policy_rejects_missing_actor(
    monkeypatch,
    tmp_path,
):
    with pytest.raises(
        ValueError,
        match="missing actors",
    ):
        build_policy(
            monkeypatch,
            tmp_path,
            selected_action=1,
            missing_agent="station",
        )
