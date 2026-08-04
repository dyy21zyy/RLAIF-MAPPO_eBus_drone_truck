from __future__ import annotations

import torch

from rlaif import reward_model_trainer


def test_cuda_context_is_initialized_before_peak_memory_reset(
    monkeypatch,
) -> None:
    calls: list[object] = []

    monkeypatch.setattr(
        torch.cuda,
        "init",
        lambda: calls.append("init"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: calls.append(("reset", str(device))),
    )

    reward_model_trainer._reset_peak_memory_stats(
        torch.device("cuda:0")
    )

    assert calls == [
        "init",
        ("reset", "cuda:0"),
    ]


def test_cpu_device_does_not_initialize_cuda(
    monkeypatch,
) -> None:
    calls: list[object] = []

    monkeypatch.setattr(
        torch.cuda,
        "init",
        lambda: calls.append("init"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: calls.append(("reset", str(device))),
    )

    reward_model_trainer._reset_peak_memory_stats(
        torch.device("cpu")
    )

    assert calls == []
