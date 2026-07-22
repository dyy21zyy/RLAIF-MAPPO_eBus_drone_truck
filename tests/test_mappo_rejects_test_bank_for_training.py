import json, pytest
from pathlib import Path
from data_pipeline.scenario_seeds import derive_scenario_seed_tuple

def test_required_module_imports():
    assert derive_scenario_seed_tuple(1).base_seed == 1
import json
from training.scenario_bank_environment import ScenarioBankEnvironmentFactory

def test_factory_rejects_test_split(tmp_path):
    p=tmp_path/'scenario_bank_manifest.json'; p.write_text(json.dumps({'schema_version':3,'split':'test','bank':'test','bank_hash':'h','scenarios':[]}))
    with pytest.raises(ValueError): ScenarioBankEnvironmentFactory(p, expected_split='test')
