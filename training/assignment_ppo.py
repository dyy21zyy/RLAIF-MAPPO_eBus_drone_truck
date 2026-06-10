"""Masked categorical actor-critic for Stage 6 assignment decisions only."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

MASKED_LOGIT = -1e9


def _mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, int(width)), nn.ReLU()))
        previous = int(width)
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class AssignmentActorCritic(nn.Module):
    """Independent actor and critic MLPs with no centralized critic or bus head."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.actor = _mlp(self.obs_dim, self.hidden_dims, self.action_dim)
        self.critic = _mlp(self.obs_dim, self.hidden_dims, 1)
        self.all_zero_mask_count = 0

    def _distribution(self, obs: Tensor, action_mask: Tensor) -> tuple[Categorical, Tensor]:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
        if obs.shape[-1] != self.obs_dim or action_mask.shape[-1] != self.action_dim:
            raise ValueError("Observation or action-mask dimension does not match the model")
        mask = action_mask.to(device=obs.device, dtype=torch.bool)
        fallback_rows = ~mask.any(dim=-1)
        if fallback_rows.any():
            # Stage 3's deterministic assignment fallback is action 0 (TD). This path
            # is explicit and counted; normal environment masks always include TD.
            mask = mask.clone()
            mask[fallback_rows, 0] = True
            self.all_zero_mask_count += int(fallback_rows.sum().item())
        logits = self.actor(obs.float()).masked_fill(~mask, MASKED_LOGIT)
        return Categorical(logits=logits), fallback_rows

    def act(
        self, obs: Tensor | Sequence[float], action_mask: Tensor | Sequence[bool], deterministic: bool = False
    ) -> tuple[int, float, float, bool]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool)
        with torch.no_grad():
            distribution, fallback_rows = self._distribution(obs_tensor, mask_tensor)
            action = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
            value = self.critic(obs_tensor.reshape(1, -1)).squeeze(-1)
            log_prob = distribution.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item()), bool(fallback_rows.item())

    def evaluate_actions(
        self, obs_batch: Tensor, action_batch: Tensor, action_mask_batch: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution, _fallback = self._distribution(obs_batch, action_mask_batch)
        actions = action_batch.long().reshape(-1)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.critic(obs_batch.float()).squeeze(-1)
        return log_prob, entropy, values
