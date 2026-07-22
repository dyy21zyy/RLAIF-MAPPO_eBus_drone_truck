"""Decentralized masked actors and the shared centralized critic for Stage 7."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


from training.event_schema import EVENT_SCHEMA_VERSION as EVENT_EMBEDDING_SCHEMA_VERSION, EVENT_NAME_TO_ID, decision_event_id
EVENT_TYPES = tuple(EVENT_NAME_TO_ID)
BUS_OPERATION_TYPES = ("loading", "charging")

def event_embedding(event_type: str, *, bus_operation: str | None = None) -> list[float]:
    """Explicit event embedding; bus loading and charging occupy different slots."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unknown event type: {event_type}")
    values = [1.0 if event_type == item else 0.0 for item in EVENT_TYPES]
    if event_type == "BUS_TERMINAL_DEPARTURE":
        bus_operation = "loading"
    elif event_type == "BUS_STATION_ARRIVAL":
        bus_operation = "charging"
    values.extend([1.0 if bus_operation == item else 0.0 for item in BUS_OPERATION_TYPES])
    return values

MASKED_LOGIT = -1e9
BUS_CHARGING_ACTIONS = (0, 15, 30, 45, 60, 75, 90, 105, 120)


def _mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("Network dimensions must be positive")
    layers: list[nn.Module] = []
    previous = int(input_dim)
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, int(width)), nn.ReLU()))
        previous = int(width)
    layers.append(nn.Linear(previous, int(output_dim)))
    return nn.Sequential(*layers)


class MaskedCategoricalActor(nn.Module):
    """Base categorical actor that makes masked actions exactly unavailable."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.policy = _mlp(self.obs_dim, self.hidden_dims, self.action_dim)

    def distribution(self, observations: Tensor, action_masks: Tensor) -> Categorical:
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        if action_masks.ndim == 1:
            action_masks = action_masks.unsqueeze(0)
        if observations.shape[-1] != self.obs_dim or action_masks.shape[-1] != self.action_dim:
            raise ValueError("Observation or action-mask dimension does not match actor")
        masks = action_masks.to(device=observations.device, dtype=torch.bool)
        if (~masks.any(dim=-1)).any():
            raise ValueError("Action mask must contain at least one feasible action")
        logits = self.policy(observations.float()).masked_fill(~masks, MASKED_LOGIT)
        return Categorical(logits=logits)

    def act(self, observation, action_mask, *, deterministic: bool = False) -> tuple[int, float]:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool)
        with torch.no_grad():
            distribution = self.distribution(observation_tensor, mask_tensor)
            action = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
            log_prob = distribution.log_prob(action)
        return int(action.item()), float(log_prob.item())

    def evaluate_actions(self, observations: Tensor, actions: Tensor, action_masks: Tensor) -> tuple[Tensor, Tensor]:
        distribution = self.distribution(observations, action_masks)
        actions = actions.long().reshape(-1)
        return distribution.log_prob(actions), distribution.entropy()


class AssignmentActor(MaskedCategoricalActor):
    """Assignment actor with action IDs ``0..2H`` (dimension ``1 + 2H``)."""

    def __init__(self, obs_dim: int, station_count: int, hidden_dims: Sequence[int] = (256, 256)) -> None:
        self.station_count = int(station_count)
        super().__init__(obs_dim, 1 + 2 * self.station_count, hidden_dims)


class BusActor(MaskedCategoricalActor):
    """Bus charging actor over the nine configured charging durations."""

    def __init__(self, obs_dim: int, hidden_dims: Sequence[int] = (256, 256)) -> None:
        super().__init__(obs_dim, len(BUS_CHARGING_ACTIONS), hidden_dims)


class CandidateScoringActor(nn.Module):
    """Masked actor that scores ``[observation, learned_event_embedding, candidate]``."""

    def __init__(
        self, obs_dim: int, candidate_feature_dim: int, hidden_dims: Sequence[int] = (256, 256), *, event_embedding_dim: int = 16
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.event_embedding_dim = int(event_embedding_dim)
        if self.event_embedding_dim <= 0:
            raise ValueError("event_embedding_dim must be a positive integer")
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.event_embedding = nn.Embedding(len(EVENT_TYPES), self.event_embedding_dim)
        self.scorer = _mlp(self.obs_dim + self.event_embedding_dim + self.candidate_feature_dim, self.hidden_dims, 1)

    def forward(self, observations: Tensor, event_type_ids: Tensor, candidate_features: Tensor, action_masks: Tensor | None = None) -> Tensor:
        if observations.ndim == 1: observations = observations.unsqueeze(0)
        if candidate_features.ndim == 2: candidate_features = candidate_features.unsqueeze(0)
        event_type_ids = event_type_ids.long().reshape(-1).to(device=observations.device)
        if observations.shape[-1] != self.obs_dim: raise ValueError("Observation dimension does not match actor")
        if candidate_features.shape[-1] != self.candidate_feature_dim: raise ValueError("Candidate-feature dimension does not match actor")
        if observations.shape[0] != candidate_features.shape[0] or observations.shape[0] != event_type_ids.shape[0]: raise ValueError("Batch length mismatch for observations, event ids, and candidates")
        if torch.any((event_type_ids < 0) | (event_type_ids >= len(EVENT_TYPES))): raise ValueError("event_type_ids must be in [0, 4]")
        repeated_obs = observations.float().unsqueeze(1).expand(-1, candidate_features.shape[1], -1)
        event_features = self.event_embedding(event_type_ids).unsqueeze(1).expand(-1, candidate_features.shape[1], -1)
        inputs = torch.cat((repeated_obs, event_features, candidate_features.float()), dim=-1)
        logits = self.scorer(inputs).squeeze(-1)
        if action_masks is not None:
            if action_masks.ndim == 1: action_masks = action_masks.unsqueeze(0)
            if candidate_features.shape[:2] != action_masks.shape[:2]: raise ValueError("Candidate features and action mask must align")
            masks = action_masks.to(device=logits.device, dtype=torch.bool)
            if (~masks.any(dim=-1)).any(): raise ValueError("Action mask must contain at least one feasible action")
            logits = logits.masked_fill(~masks, MASKED_LOGIT)
        return logits

    def distribution(self, observations: Tensor, event_type_ids: Tensor, candidate_features: Tensor | None = None, action_masks: Tensor | None = None) -> Categorical:
        if action_masks is None:  # legacy distribution(observations, candidates, masks)
            action_masks = candidate_features
            candidate_features = event_type_ids
            event_type_ids = torch.zeros(observations.shape[0] if observations.ndim > 1 else 1, dtype=torch.long, device=observations.device)
        return Categorical(logits=self.forward(observations, event_type_ids, candidate_features, action_masks))

    def act(self, observation, event_type_id: int, candidate_features=None, action_mask=None, *, deterministic: bool = False) -> tuple[int, float]:
        if action_mask is None:  # legacy boundary: infer PARCEL_RELEASE for old tests/callers.
            action_mask = candidate_features
            candidate_features = event_type_id
            event_type_id = 0
        with torch.no_grad():
            obs = torch.as_tensor(observation, dtype=torch.float32)
            candidates = torch.as_tensor(candidate_features, dtype=torch.float32)
            mask = torch.as_tensor(action_mask, dtype=torch.bool)
            event_ids = torch.tensor([int(event_type_id)], dtype=torch.long)
            distribution = self.distribution(obs, event_ids, candidates, mask)
            action = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
            log_prob = distribution.log_prob(action)
        return int(action.item()), float(log_prob.item())

    def evaluate_actions(self, observations: Tensor, event_type_ids: Tensor, candidate_features: Tensor | None = None, action_masks: Tensor | None = None, selected_actions: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if selected_actions is None:  # legacy evaluate_actions(observations, candidates, actions, masks)
            selected_actions = candidate_features
            candidate_features = event_type_ids
            action_masks = action_masks
            event_type_ids = torch.zeros(observations.shape[0] if observations.ndim > 1 else 1, dtype=torch.long, device=observations.device)
        distribution = self.distribution(observations, event_type_ids, candidate_features, action_masks)
        actions = selected_actions.long().reshape(-1)
        return distribution.log_prob(actions), distribution.entropy()

def build_actor_registry(
    specs: dict[str, tuple[int, int]], hidden_dims: dict[str, Sequence[int]],
    event_embedding_dim: int = 16,
) -> nn.ModuleDict:
    """Build the Stage 9 four-agent candidate-scoring actor registry."""

    required = {"assignment", "truck", "bus", "station"}
    if set(specs) != required:
        raise ValueError(f"Actor specs must cover exactly {sorted(required)}")
    default_hidden = tuple(hidden_dims.get("default", (256, 256)))
    actors = {
        agent_id: CandidateScoringActor(
            obs_dim=obs_dim,
            candidate_feature_dim=candidate_dim,
            hidden_dims=hidden_dims.get(agent_id, default_hidden),
            event_embedding_dim=event_embedding_dim,
        )
        for agent_id, (obs_dim, candidate_dim) in specs.items()
    }
    return nn.ModuleDict(actors)


class CentralizedCritic(nn.Module):
    """Shared value function over the environment's global state."""

    def __init__(self, global_state_dim: int, hidden_dims: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        self.global_state_dim = int(global_state_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)
        self.value_network = _mlp(self.global_state_dim, self.hidden_dims, 1)

    def forward(self, global_states: Tensor) -> Tensor:
        if global_states.ndim == 1:
            global_states = global_states.unsqueeze(0)
        if global_states.shape[-1] != self.global_state_dim:
            raise ValueError("Global-state dimension does not match critic")
        return self.value_network(global_states.float()).squeeze(-1)
