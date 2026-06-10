"""Decentralized masked actors and the shared centralized critic for Stage 7."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

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
