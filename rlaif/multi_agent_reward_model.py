from __future__ import annotations
from typing import Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F
class MultiAgentRewardModel(nn.Module):
    def __init__(self,state_dim:int,candidate_dim:int,event_types:Sequence[str],hidden_dims:Sequence[int]=(64,64),dropout:float=0.0):
        super().__init__(); self.event_types=tuple(event_types); self.event_to_id={e:i for i,e in enumerate(self.event_types)}; self.event_embedding=nn.Embedding(max(1,len(self.event_types)),4)
        layers=[]; d=state_dim+candidate_dim+4
        for h in hidden_dims:
            layers += [nn.Linear(d,int(h)), nn.ReLU()];
            if dropout: layers.append(nn.Dropout(dropout))
            d=int(h)
        layers.append(nn.Linear(d,1)); self.mlp=nn.Sequential(*layers)
    def forward(self,state:Tensor,candidate:Tensor,event_type_ids:Tensor)->Tensor:
        return self.mlp(torch.cat([state.float(),candidate.float(),self.event_embedding(event_type_ids.long())],-1)).squeeze(-1)
def bradley_terry_loss(chosen:Tensor,rejected:Tensor)->Tensor: return -F.logsigmoid(chosen-rejected).mean()
