import pytest
import torch

from experiments.estimate_reward_reference_scales import classify_and_estimate, RewardScaleEstimationError
from experiments.analyze_hard_locker_paired_results import _metric
from training.device import resolve_torch_device


def _rows(value=0.0):
    from envs.reward_components import REWARD_COMPONENTS
    row={"episode_status":"success"}
    row.update({f"raw_{name}": (value if name == "locker_overflow" else 1.0) for name in REWARD_COMPONENTS})
    return [row]


def test_structural_zero_schema_and_positive_rejection():
    cfg={"structural_zero_components":{"locker_overflow":{"positive_denominator":1.0,"reason":"hard capacity"}}}
    components, scales, _=classify_and_estimate(_rows(),cfg)
    locker=components["locker_overflow"]
    assert locker["structural_zero"] is True
    assert locker["positive_count"] == 0
    assert locker["denominator"] == locker["scale"] == scales["locker_overflow"] == 1.0
    assert locker["denominator_source"] == "explicit_structural_fallback"
    with pytest.raises(RewardScaleEstimationError,match="positive samples"):
        classify_and_estimate(_rows(0.1),cfg)


def test_formal_metric_unwrap_is_strict():
    assert _metric({"formal_metrics":{"fulfillment_rate":{"value":.5,"available":True}}},"fulfillment_rate") == .5
    for record in ({"value":1,"available":False}, 1, {"available":True}):
        with pytest.raises(ValueError): _metric({"formal_metrics":{"fulfillment_rate":record}},"fulfillment_rate")
    with pytest.raises(ValueError): _metric({"formal_metrics":{}},"fulfillment_rate")


def test_device_policy_cpu_and_fail_closed(monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    assert resolve_torch_device("auto") == torch.device("cpu")
    with pytest.raises(RuntimeError): resolve_torch_device("auto",require_cuda=True)
    with pytest.raises(RuntimeError): resolve_torch_device("cuda")
    with pytest.raises(ValueError): resolve_torch_device("tpu")


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_real_cuda_tensor_smoke():
    device=resolve_torch_device("cuda",require_cuda=True)
    assert (torch.ones(2,2,device=device) @ torch.ones(2,2,device=device)).device == device
