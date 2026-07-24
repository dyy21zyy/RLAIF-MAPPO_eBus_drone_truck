import torch
from torch import nn

from training.mappo_buffer import AsyncMAPPOBuffer
from training.mappo_trainer import collect_episode
from training.reward_contribution import RewardContribution
from rlaif.reward_registry import RewardRegistry
from envs.state_builder import BUS_LOADING_FEATURE_NAMES, BUS_CHARGING_FEATURE_NAMES


CANDIDATE_NAMES = ("action_type_id", "estimated_time_norm")


class RecordingActor(nn.Module):
    def __init__(self, obs_dim, action=1):
        super().__init__()
        self.obs_dim = obs_dim
        self.action = action
        self.seen = []

    def act(self, observation, event_type_id, candidate_features, mask, deterministic=False):
        self.seen.append({
            "observation": list(observation),
            "event_type_id": event_type_id,
            "candidate_features": [list(row) for row in candidate_features],
            "mask": list(mask),
        })
        return self.action, -0.25


class ConstantCritic(nn.Module):
    def forward(self, state):
        return torch.tensor(0.0)


class RecordingRegistry(RewardRegistry):
    def __init__(self):
        self.calls = []

    def score_transition(self, **kwargs):
        self.calls.append(kwargs)
        return RewardContribution(
            kwargs["agent_type"], kwargs["event_type"], kwargs["environment_reward"],
            0.5, 0.5, 0.5, 1.0, 0.5, kwargs["environment_reward"] + 0.5, False, "test"
        )


class RecordingLegacyWrapper:
    enabled = True

    def __init__(self):
        self.calls = []

    def score(self, state_features, action_features, action_id, event_type=None):
        self.calls.append({
            "state_features": list(state_features),
            "action_features": list(action_features),
            "action_id": action_id,
            "event_type": event_type,
        })
        return 0.25


class SequenceEnv:
    def __init__(self, observations):
        self.observations = list(observations)
        self.index = 0
        self.parcels = {}
        self.cost_components = {}
        self.infeasible_action_corrections = 0
        self.decision_counts = {"assignment": 0, "truck": 0, "bus": 0, "station": 0}

    def reset(self, seed=None):
        self.index = 0
        return self.observations[0], {}

    def step(self, action):
        agent = self.observations[self.index]["agent_id"]
        self.decision_counts[agent] += 1
        self.index += 1
        obs = self.observations[self.index] if self.index < len(self.observations) else terminal_obs()
        return obs, 1.0, obs["agent_id"] == "terminal", False, {"reward_components": {}}

    def get_entity_critic_state(self):
        return [0.0, 1.0, 2.0]


def terminal_obs():
    return {"agent_id": "terminal"}


def obs(agent_id, event_type, features, candidates=None, action=1):
    candidates = candidates or [[10.0, 11.0], [20.0, 21.0]]
    return {
        "agent_id": agent_id,
        "event_type": event_type,
        "features": list(features),
        "action_mask": [True] * len(candidates),
        "candidate_actions": [{"action_type": "noop"}, {"action_type": "selected"}],
        "candidate_features": [list(row) for row in candidates],
        "candidate_feature_names": CANDIDATE_NAMES,
        "time_min": 12.0,
    }


def actor_registry(bus_actor):
    return nn.ModuleDict({
        "assignment": RecordingActor(5),
        "truck": RecordingActor(4),
        "bus": bus_actor,
        "station": RecordingActor(16),
    })


def test_bus_departure_reward_gets_raw_7_and_actor_buffer_get_padded_8():
    raw = [float(i) for i in range(len(BUS_LOADING_FEATURE_NAMES))]
    selected_candidate = [20.0, 21.0]
    bus_actor = RecordingActor(obs_dim=8)
    buffer = AsyncMAPPOBuffer()
    registry = RecordingRegistry()

    collect_episode(SequenceEnv([obs("bus", "BUS_TERMINAL_DEPARTURE", raw), terminal_obs()]), actor_registry(bus_actor), ConstantCritic(), buffer, registry, episode_id=1, lambda_rlaif=1.0, deterministic=True)

    assert len(raw) == 7
    assert len(bus_actor.seen[0]["observation"]) == 8
    assert bus_actor.seen[0]["observation"] == raw + [0.0]
    assert len(registry.calls[0]["state_features"]) == 7
    assert registry.calls[0]["state_features"] == raw
    assert registry.calls[0]["candidate_features"] == selected_candidate
    assert len(buffer.transitions[0].local_obs) == 8
    assert buffer.transitions[0].local_obs == raw + [0.0]


def test_bus_arrival_reward_and_actor_get_raw_8_without_candidate_mutation():
    raw = [float(i) / 10.0 for i in range(len(BUS_CHARGING_FEATURE_NAMES))]
    candidates = [[1.0, 2.0], [3.0, 4.0]]
    bus_actor = RecordingActor(obs_dim=8)
    registry = RecordingRegistry()

    collect_episode(SequenceEnv([obs("bus", "BUS_STATION_ARRIVAL", raw, candidates), terminal_obs()]), actor_registry(bus_actor), ConstantCritic(), AsyncMAPPOBuffer(), registry, episode_id=2, lambda_rlaif=1.0, deterministic=True)

    assert len(raw) == 8
    assert len(bus_actor.seen[0]["observation"]) == 8
    assert registry.calls[0]["state_features"] == raw
    assert len(registry.calls[0]["state_features"]) == 8
    assert registry.calls[0]["candidate_features"] == candidates[1]


def test_legacy_reward_wrapper_gets_raw_unpadded_decision_surface_features():
    raw = [float(i) for i in range(len(BUS_LOADING_FEATURE_NAMES))]
    bus_actor = RecordingActor(obs_dim=8)
    wrapper = RecordingLegacyWrapper()

    collect_episode(SequenceEnv([obs("bus", "BUS_TERMINAL_DEPARTURE", raw), terminal_obs()]), actor_registry(bus_actor), ConstantCritic(), AsyncMAPPOBuffer(), wrapper, episode_id=3, lambda_rlaif=1.0, deterministic=True)

    assert bus_actor.seen[0]["observation"] == raw + [0.0]
    assert wrapper.calls[0]["state_features"] == raw
    assert wrapper.calls[0]["action_features"] == [20.0, 21.0]


def test_assignment_truck_and_station_transitions_still_collect():
    observations = [
        obs("assignment", "PARCEL_RELEASE", [0.0] * 5),
        obs("truck", "TRUCK_AVAILABLE", [0.0] * 4),
        obs("station", "STATION_OPERATION", [0.0] * 16),
        terminal_obs(),
    ]
    buffer = AsyncMAPPOBuffer()
    registry = RecordingRegistry()

    collect_episode(SequenceEnv(observations), actor_registry(RecordingActor(8)), ConstantCritic(), buffer, registry, episode_id=4, lambda_rlaif=1.0, deterministic=True)

    assert [t.agent_id for t in buffer.transitions] == ["assignment", "truck", "station"]
    assert [c["agent_type"] for c in registry.calls] == ["assignment", "truck", "station"]


def test_short_bus_integration_with_both_event_types_has_no_mismatch_or_fallback():
    observations = [
        obs("bus", "BUS_TERMINAL_DEPARTURE", [0.0] * len(BUS_LOADING_FEATURE_NAMES)),
        obs("bus", "BUS_STATION_ARRIVAL", [0.0] * len(BUS_CHARGING_FEATURE_NAMES)),
        terminal_obs(),
    ]
    buffer = AsyncMAPPOBuffer()
    registry = RecordingRegistry()

    collect_episode(SequenceEnv(observations), actor_registry(RecordingActor(8)), ConstantCritic(), buffer, registry, episode_id=5, lambda_rlaif=1.0, deterministic=True)

    assert [len(c["state_features"]) for c in registry.calls] == [7, 8]
    assert all(not t.used_rlaif_fallback for t in buffer.transitions)
    assert [len(t.local_obs) for t in buffer.transitions] == [8, 8]
