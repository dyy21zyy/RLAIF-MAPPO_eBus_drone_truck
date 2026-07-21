"""Learned Stage 5 assignment reward model and pairwise objective."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class AssignmentRewardModel(nn.Module):
    """Score one normalized assignment state/action pair with a scalar reward."""

    def __init__(self, state_dim: int, action_feature_dim: int, num_actions: int,
                 action_emb_dim: int = 16, hidden_dims: Sequence[int] = (256, 256, 128),
                 dropout: float = 0.0) -> None:
        super().__init__()
        if min(state_dim, action_feature_dim, num_actions, action_emb_dim) <= 0:
            raise ValueError("model dimensions must be positive")
        self.action_embedding = nn.Embedding(num_actions, action_emb_dim)
        layers: list[nn.Module] = []
        input_dim = state_dim + action_feature_dim + action_emb_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, int(hidden_dim)), nn.ReLU()))
            if dropout:
                layers.append(nn.Dropout(dropout))
            input_dim = int(hidden_dim)
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, state_features: Tensor, action_features: Tensor, action_ids: Tensor) -> Tensor:
        embedding = self.action_embedding(action_ids.long())
        combined = torch.cat((state_features.float(), action_features.float(), embedding), dim=-1)
        return self.mlp(combined).squeeze(-1)


def pairwise_preference_loss(chosen_scores: Tensor, rejected_scores: Tensor) -> Tensor:
    return -F.logsigmoid(chosen_scores - rejected_scores).mean()


def normalize_reward(scores: Tensor, reward_mean: float | Tensor, reward_std: float | Tensor,
                     epsilon: float = 1e-6) -> Tensor:
    return (scores - torch.as_tensor(reward_mean, dtype=scores.dtype, device=scores.device)) / (
        torch.as_tensor(reward_std, dtype=scores.dtype, device=scores.device) + epsilon
    )


# Phase 8 alias; independent multi-agent models are defined without action-ID embeddings.
from rlaif.multi_agent_reward_model import MultiAgentRewardModel, bradley_terry_loss
