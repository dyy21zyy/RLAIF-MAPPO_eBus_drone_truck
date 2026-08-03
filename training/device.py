"""Shared, fail-closed torch device resolution."""
from __future__ import annotations
import torch

def resolve_torch_device(requested_device: str = "auto", *, require_cuda: bool = False) -> torch.device:
    """Resolve a supported torch device without an implicit required-CUDA fallback."""
    value = str(requested_device or "auto").strip().lower()
    if value == "auto": value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cpu":
        if require_cuda: raise RuntimeError("CUDA is required; CPU fallback is disabled")
        return torch.device("cpu")
    if value == "cuda" or value.startswith("cuda:"):
        if value.startswith("cuda:") and not value.partition(":")[2].isdigit():
            raise ValueError("CUDA device must use a non-negative integer index, for example cuda:0")
        if not torch.cuda.is_available(): raise RuntimeError(f"requested device {value!r}, but CUDA is unavailable")
        parsed_device = torch.device(value)
        index = torch.cuda.current_device() if parsed_device.index is None else parsed_device.index
        if index < 0 or index >= torch.cuda.device_count(): raise RuntimeError(f"CUDA device index {index} is out of range")
        return torch.device("cuda", index)
    raise ValueError("device must be one of auto, cpu, cuda, or cuda:<index>")
