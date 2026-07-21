"""Stage 6 masked policy, configuration, and smoke-test tests."""

from pathlib import Path

import pytest

from tests.torch_optional import import_optional_torch
from utils.config import load_config

ROOT = Path(__file__).parents[1]


def test_assignment_ppo_config_loads() -> None:
    config = load_config(ROOT / "configs/train_assignment_ppo.yaml")
    assert config["rlaif"]["enabled"] is False
    assert config["bus_baseline"]["name"] == "uniform_30"
    assert config["policy"]["hidden_dims"] == [256, 256]


def test_masked_distribution_never_samples_infeasible_action() -> None:
    torch = import_optional_torch()
    from training.assignment_ppo import AssignmentActorCritic

    model = AssignmentActorCritic(3, 5, [8, 8])
    for _ in range(200):
        action, *_ = model.act(torch.zeros(3), torch.tensor([False, True, False, False, True]))
        assert action in {1, 4}


def test_all_zero_assignment_mask_uses_explicit_td_fallback() -> None:
    torch = import_optional_torch()
    from training.assignment_ppo import AssignmentActorCritic

    model = AssignmentActorCritic(2, 3, [8])
    action, _log_prob, _value, fallback = model.act(torch.zeros(2), torch.zeros(3, dtype=torch.bool))
    assert action == 0
    assert fallback is True
    assert model.all_zero_mask_count == 1


def test_smoke_test_skips_cleanly_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    import experiments.smoke_test_assignment_ppo as smoke

    monkeypatch.setattr(smoke, "is_torch_runtime_available", lambda: False)
    assert smoke.run_smoke_test() == {"skipped": True, "reason": "PyTorch is unavailable"}


def test_assignment_rollout_stores_no_bus_transitions(tmp_path: Path) -> None:
    import_optional_torch()
    from training.assignment_ppo import AssignmentActorCritic
    from training.bus_baseline_policy import BusBaselinePolicy
    from training.ppo_buffer import PPOBuffer
    from training.ppo_trainer import collect_episode, create_environment
    from training.reward_model_wrapper import RewardModelWrapper

    config = load_config(ROOT / "configs/train_assignment_ppo.yaml")
    env = create_environment(config, output_root=tmp_path)
    observation, _ = env.reset()
    model = AssignmentActorCritic(len(observation["features"]), env.assignment_action_size, [8, 8])
    buffer = PPOBuffer()
    metrics = collect_episode(
        env, model, buffer, BusBaselinePolicy("no_charge"), RewardModelWrapper(None, enabled=False),
        episode_id=1, lambda_rlaif=1.0,
    )
    assert len(buffer) == metrics["assignment_decision_count"] > 0
    assert all(item.info["agent"] == "assignment" for item in buffer.transitions)
    assert all(item.action_mask[item.action] for item in buffer.transitions)
