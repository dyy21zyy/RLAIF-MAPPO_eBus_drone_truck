import json, pytest
from pathlib import Path
from data_pipeline.scenario_seeds import derive_scenario_seed_tuple

def test_required_module_imports():
    assert derive_scenario_seed_tuple(1).base_seed == 1
from experiments.build_scenario_bank import build_bank

def test_formal_generation_rejects_fallback(tmp_path):
    with pytest.raises(ValueError):
        build_bank('configs/shanghai_small.yaml','train',1,1,tmp_path,fallback=True,run_classification='formal')
