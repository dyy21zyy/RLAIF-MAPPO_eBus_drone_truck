import json, pytest
from pathlib import Path
from data_pipeline.scenario_seeds import derive_scenario_seed_tuple

def test_required_module_imports():
    assert derive_scenario_seed_tuple(1).base_seed == 1
