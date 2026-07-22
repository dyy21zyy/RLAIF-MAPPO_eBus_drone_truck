import pytest
from rlaif.preference_schema_v3 import validate_label_source

def test_external_evaluator_accepted_formally(): validate_label_source('external_evaluator_api',formal_mode=True)
def test_validated_replay_accepted_formally(): validate_label_source('validated_replay',formal_mode=True)
def test_synthetic_smoke_only_smoke_mode():
    validate_label_source('synthetic_smoke',formal_mode=False)
    with pytest.raises(ValueError): validate_label_source('synthetic_smoke',formal_mode=True)
def test_unknown_source_fails():
    with pytest.raises(ValueError): validate_label_source('unknown',formal_mode=False)
