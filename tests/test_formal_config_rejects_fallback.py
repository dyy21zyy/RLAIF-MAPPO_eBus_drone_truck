import pytest
from training.config_resolver import validate_run_classification, TrainingConfigError


def test_formal_fallback_true_fails():
    with pytest.raises(TrainingConfigError, match='env.fallback'):
        validate_run_classification({'run_classification':'formal','env':{'fallback':True}})


def test_smoke_explicit_fallback_passes():
    validate_run_classification({'run_classification':'smoke','env':{'fallback':True}})


def test_formal_rlaif_fallback_fails():
    with pytest.raises(TrainingConfigError, match='fallback_to_env_reward'):
        validate_run_classification({'run_classification':'formal','env':{'fallback':False},'rlaif':{'fallback_to_env_reward':True}})
