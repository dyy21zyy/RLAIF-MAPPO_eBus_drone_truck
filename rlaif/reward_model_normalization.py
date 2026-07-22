from __future__ import annotations
from dataclasses import dataclass
import torch
@dataclass(frozen=True)
class FeatureNormalization:
    mean: tuple[float,...]; std: tuple[float,...]; feature_names: tuple[str,...]
def fit_feature_normalization(tensor: torch.Tensor, *, feature_names: tuple[str,...]) -> FeatureNormalization:
    if tensor.ndim!=2 or tensor.shape[1]!=len(feature_names): raise ValueError('normalization tensor shape does not match feature names')
    mean=tensor.float().mean(dim=0); std=tensor.float().std(dim=0, unbiased=False); std=torch.where(std==0, torch.ones_like(std), std)
    return FeatureNormalization(tuple(float(x) for x in mean), tuple(float(x) for x in std), tuple(feature_names))
def apply_feature_normalization(tensor: torch.Tensor, norm: FeatureNormalization, *, feature_names: tuple[str,...]) -> torch.Tensor:
    if tuple(feature_names)!=norm.feature_names: raise ValueError('feature-name order differs from fitted normalization')
    return (tensor.float()-torch.tensor(norm.mean,dtype=torch.float32))/torch.tensor(norm.std,dtype=torch.float32)
