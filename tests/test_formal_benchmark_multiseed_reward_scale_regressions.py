from types import SimpleNamespace

import evaluation.formal_episode_runner as episode_runner
from evaluation.preformal_part3_gates import validate_expected_rows


def test_formal_episode_uses_frozen_reward_scale_before_environment_reset(
    monkeypatch,
    tmp_path,
):
    observed_reward_config = {}

    class FakeEnvironment:
        def __init__(self, instance_path):
            self.instance_path = instance_path
            self.config = {
                "run_classification": "formal",
                "reward": {
                    "apply_reference_scales": True,
                    "scale_artifact": (
                        "outputs/reward_reference_scales_v1.json"
                    ),
                    "scale_artifact_hash": (
                        "REPLACE_WITH_REAL_SCALE_HASH"
                    ),
                },
            }

        def reset(self, *, seed=None, options=None):
            del seed, options
            observed_reward_config.update(self.config["reward"])
            return {"agent_id": "assignment"}, {}

        def step(self, action):
            del action
            return (
                {"agent_id": "terminal"},
                0.0,
                True,
                False,
                {},
            )

        def check_invariants(self):
            return []

    class FakePolicy:
        def select_action(self, **kwargs):
            del kwargs
            return 0

    monkeypatch.setattr(
        episode_runner,
        "DynamicDeliveryEnv",
        FakeEnvironment,
    )
    monkeypatch.setattr(
        episode_runner,
        "load_frozen_instance",
        lambda scenario: scenario,
    )
    monkeypatch.setattr(
        episode_runner,
        "collect_formal_metrics",
        lambda env, runtime_seconds, transition_count, rlaif: ({}, {}),
    )
    monkeypatch.setattr(
        episode_runner,
        "validate_formal_metrics",
        lambda metrics: None,
    )

    frozen_scale = tmp_path / "final_reward_reference_scales.json"

    result = episode_runner.evaluate_policy_on_frozen_scenario(
        scenario=SimpleNamespace(
            instance_path=tmp_path / "instance.json",
        ),
        method_spec=SimpleNamespace(),
        policy=FakePolicy(),
        reward_registry=None,
        evaluation_config={
            "reward_scale_artifact_path": str(frozen_scale),
            "reward_scale_artifact_hash": "frozen-scale-hash",
        },
        training_seed=1,
    )

    assert result.status == "success"
    assert (
        observed_reward_config["scale_artifact"]
        == str(frozen_scale)
    )
    assert (
        observed_reward_config["scale_artifact_hash"]
        == "frozen-scale-hash"
    )


def test_expected_rows_inherit_top_level_training_seeds():
    methods = [
        {"method_id": "mappo_env"},
        {"method_id": "mappo_rlaif_assignment"},
    ]
    scenarios = ["test_0000", "test_0001"]

    report = validate_expected_rows(
        methods,
        scenarios,
        [],
        training_seeds=[1, 2, 3],
    )

    assert report["expected_rows"] == 12
    assert {
        identity[1]
        for identity in report["expected_identities"]
    } == {1, 2, 3}
