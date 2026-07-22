import pytest
from training.config_resolver import validate_run_classification, TrainingConfigError


def test_positive_truck_reward_zero_coefficients_fails():
    c={'run_classification':'formal','env':{'fallback':False},'reward':{'truck_cost':1}}
    with pytest.raises(TrainingConfigError, match='positive weight to truck_cost'):
        validate_run_classification(c, config_only=True)


def test_positive_distance_coefficient_passes():
    c={'run_classification':'formal','env':{'fallback':False},'reward':{'truck_cost':1},'truck':{'cost_per_km':1}}
    validate_run_classification(c, config_only=True)
