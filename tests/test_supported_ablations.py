import pytest
from experiments.run_paper_ablation import validate_ablations

def test_unsupported_ablations_fail_validation():
    with pytest.raises(ValueError): validate_ablations({"ablations":[{"name":"fake","config_switch":"x","checkpoint":"c.pt"}]})

def test_supported_ablation_requires_switch_and_checkpoint():
    with pytest.raises(ValueError): validate_ablations({"ablations":[{"name":"no_rlaif"}]})
