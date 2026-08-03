import pytest
import torch

from training.device import resolve_torch_device


def test_auto_uses_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_torch_device("auto", require_cuda=False) == torch.device("cpu")


def test_required_cuda_never_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="required"):
        resolve_torch_device("auto", require_cuda=True)


def test_cpu_is_supported_and_rejected_when_cuda_required():
    assert resolve_torch_device("cpu", require_cuda=False) == torch.device("cpu")
    with pytest.raises(RuntimeError, match="fallback"):
        resolve_torch_device("cpu", require_cuda=True)


def test_invalid_values_and_cuda_index_fail_clearly(monkeypatch):
    with pytest.raises(ValueError, match="device must"):
        resolve_torch_device("mps")
    with pytest.raises(ValueError, match="integer index"):
        resolve_torch_device("cuda:x")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="out of range"):
        resolve_torch_device("cuda:1")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_auto_selects_real_cuda_and_executes_tensor_operation():
    device = resolve_torch_device("auto", require_cuda=True)
    assert device.type == "cuda"
    assert (torch.ones(2, device=device) + 1).device == device
