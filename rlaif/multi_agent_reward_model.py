from __future__ import annotations
from typing import Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F

class AgentRewardModel(nn.Module):
    def __init__(self, *, state_dim:int, candidate_dim:int, num_event_types:int, event_embedding_dim:int, hidden_dims:tuple[int,...], dropout:float) -> None:
        super().__init__()
        if min(state_dim, candidate_dim, num_event_types, event_embedding_dim) <= 0: raise ValueError('model dimensions must be positive')
        self.state_dim=int(state_dim); self.candidate_dim=int(candidate_dim); self.num_event_types=int(num_event_types); self.event_embedding_dim=int(event_embedding_dim); self.hidden_dims=tuple(int(x) for x in hidden_dims); self.dropout=float(dropout)
        self.event_embedding=nn.Embedding(self.num_event_types,self.event_embedding_dim)
        layers=[]; d=self.state_dim+self.candidate_dim+self.event_embedding_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(d,h), nn.ReLU()]
            if self.dropout: layers.append(nn.Dropout(self.dropout))
            d=h
        layers.append(nn.Linear(d,1)); self.mlp=nn.Sequential(*layers)
    def forward(self, state_features:Tensor, event_type_ids:Tensor, candidate_features:Tensor) -> Tensor:
        if state_features.shape[-1] != self.state_dim: raise ValueError('state feature dimension mismatch')
        if candidate_features.shape[-1] != self.candidate_dim: raise ValueError('candidate feature dimension mismatch')
        event_type_ids=event_type_ids.long().reshape(-1).to(state_features.device)
        if torch.any((event_type_ids < 0) | (event_type_ids >= self.num_event_types)): raise ValueError('event_type_ids out of range')
        x=torch.cat([state_features.float(), self.event_embedding(event_type_ids), candidate_features.float()], dim=-1)
        return self.mlp(x).squeeze(-1)

class MultiAgentRewardModel(AgentRewardModel):
    def __init__(self,state_dim:int,candidate_dim:int,event_types:Sequence[str],hidden_dims:Sequence[int]=(64,64),dropout:float=0.0,event_embedding_dim:int=16):
        self.event_types=tuple(event_types); self.event_to_id={e:i for i,e in enumerate(self.event_types)}
        super().__init__(state_dim=state_dim,candidate_dim=candidate_dim,num_event_types=max(1,len(self.event_types)),event_embedding_dim=event_embedding_dim,hidden_dims=tuple(hidden_dims),dropout=dropout)
    def forward(self, state_features:Tensor, arg2:Tensor, arg3:Tensor)->Tensor:
        # Backward compatible signature: (state, candidate, event_ids). New AgentRewardModel signature is (state, event_ids, candidate).
        if arg2.dtype.is_floating_point and not arg3.dtype.is_floating_point:
            return super().forward(state_features, arg3, arg2)
        return super().forward(state_features, arg2, arg3)

def bradley_terry_loss(score_a:Tensor, score_b:Tensor, target_a_preferred:Tensor|None=None)->Tensor:
    if target_a_preferred is None: target_a_preferred=torch.ones_like(score_a)
    return F.binary_cross_entropy_with_logits(score_a-score_b, target_a_preferred.float())
