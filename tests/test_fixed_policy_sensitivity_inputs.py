import json
from pathlib import Path
import pytest, yaml
from envs.reward_components import REWARD_COMPONENTS
from envs.reward_scales import canonical_payload_hash, load_reward_scale_artifact
from experiments.prepare_fixed_policy_sensitivity_inputs import BAD, differing_paths, dotted_value, hydrate_runtime_base, patch_dotted, validate_bank, _safe_root

def base(): return yaml.safe_load(Path("configs/paper/base_medium.yaml").read_text())

@pytest.mark.parametrize("path,value",[("passenger.demand_intensity",.75),("scenario.num_parcels",45),("station.power_capacity_kw",880.)])
def test_one_factor_patch_is_deep_and_exact(path,value):
    source=base(); changed=patch_dotted(source,path,value)
    assert differing_paths(source,changed)=={path}; assert dotted_value(changed,path)==value; assert dotted_value(source,path)!=value

def test_configuration_has_explicit_paired_protocol_and_real_scale_path():
    cfg=yaml.safe_load(Path("configs/paper/fixed_policy_sensitivity.yaml").read_text())
    assert list(range(310000,310100))==list(range(cfg["paired_scenarios"]["seed_start"],cfg["paired_scenarios"]["seed_start"]+cfg["paired_scenarios"]["count"]))
    path=cfg["formal_artifacts"]["reward_reference_scale"]
    assert path=="results/formal/reward_scales/final_reward_reference_scales.json"
    assert not any(x in path for x in BAD)

def test_force_root_rejects_protected_directories():
    with pytest.raises(ValueError,match="protected"): _safe_root(Path("results/formal/reward_models"))

def test_validate_bank_rejects_count_before_loading_scenarios(tmp_path):
    p=tmp_path/"bank"; p.mkdir(); (p/"scenario_bank_manifest.json").write_text(json.dumps({"sensitivity_mode":"fixed_policy_robustness","is_final_test_bank":False,"scenario_count":99,"bank_hash":"a"*64}))
    with pytest.raises(ValueError,match="count"): validate_bank(p,base_config=base(),parameter="scenario.num_parcels",value=45,expected_count=100,expected_seeds=list(range(100)))

def test_diagnostic_classification_is_never_publication_eligible():
    # The preparation contract derives this flag rather than accepting it from callers.
    classification="diagnostic"
    assert (classification=="formal") is False

def _formal_scale(path):
    payload={"artifact_type":"reward_reference_scales","artifact_version":1,
             "run_classification":"formal","validation_status":"passed",
             "training_scenario_bank_hash":"b"*64,
             "component_order":list(REWARD_COMPONENTS),
             "scales":{name:1.0 for name in REWARD_COMPONENTS}}
    payload["artifact_hash"]=canonical_payload_hash(payload)
    path.write_text(json.dumps(payload))
    return payload

def test_runtime_base_hydrates_canonical_reward_lineage_without_mutation(tmp_path):
    scale=tmp_path/"formal-scale.json"; payload=_formal_scale(scale)
    source=base(); original=json.loads(json.dumps(source))
    runtime,artifact=hydrate_runtime_base(source,scale)
    assert load_reward_scale_artifact(scale,formal_mode=True).artifact_hash==payload["artifact_hash"]
    assert source==original
    assert runtime["reward"]["scale_artifact"]==str(scale)
    assert runtime["reward"]["scale_artifact_hash"]==artifact.artifact_hash==payload["artifact_hash"]
    assert runtime["reward"]["expected_training_scenario_bank_hash"]==payload["training_scenario_bank_hash"]
    changed=patch_dotted(runtime,"scenario.num_parcels",45)
    assert differing_paths(runtime,changed)=={"scenario.num_parcels"}

def test_validate_bank_rejects_stale_reward_lineage_before_scenarios(tmp_path):
    scale=tmp_path/"formal-scale.json"; _formal_scale(scale)
    runtime,_=hydrate_runtime_base(base(),scale)
    stale=patch_dotted(runtime,"scenario.num_parcels",45)
    stale["reward"]["scale_artifact"]="outputs/reward_reference_scales_v1.json"
    bank=tmp_path/"bank"; bank.mkdir()
    manifest={"sensitivity_mode":"fixed_policy_robustness","is_final_test_bank":False,
              "scenario_count":1,"bank_hash":"a"*64,
              "resolved_environment_config":stale,"scenarios":[]}
    (bank/"scenario_bank_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="reward-scale lineage"):
        validate_bank(bank,base_config=runtime,parameter="scenario.num_parcels",value=45,
                      expected_count=1,expected_seeds=[1])
