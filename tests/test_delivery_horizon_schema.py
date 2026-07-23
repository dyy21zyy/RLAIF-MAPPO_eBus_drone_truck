
import pytest

from data_pipeline.build_instance import normalize_dynamic_config


def test_canonical_delivery_horizon_is_used_without_legacy_alias():
    cfg = {"time": {"delivery_evaluation_horizon_min": 480}, "bus": {}}
    resolved = normalize_dynamic_config(cfg)
    assert resolved["time"]["delivery_evaluation_horizon_min"] == 480
    assert resolved["bus"]["delivery_horizon_min"] == 480


def test_legacy_delivery_horizon_alias_is_normalized():
    cfg = {"time": {}, "bus": {"delivery_horizon_min": 420}}
    resolved = normalize_dynamic_config(cfg)
    assert resolved["time"]["delivery_evaluation_horizon_min"] == 420
    assert resolved["bus"]["delivery_horizon_min"] == 420


def test_equal_canonical_and_legacy_delivery_horizons_are_accepted():
    cfg = {"time": {"delivery_evaluation_horizon_min": 480}, "bus": {"delivery_horizon_min": 480}}
    resolved = normalize_dynamic_config(cfg)
    assert resolved["time"]["delivery_evaluation_horizon_min"] == 480
    assert resolved["bus"]["delivery_horizon_min"] == 480


def test_conflicting_canonical_and_legacy_delivery_horizons_fail():
    cfg = {"time": {"delivery_evaluation_horizon_min": 480}, "bus": {"delivery_horizon_min": 360}}
    with pytest.raises(ValueError, match="conflicting delivery horizon values"):
        normalize_dynamic_config(cfg)
