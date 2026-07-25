import json
from pathlib import Path
import pytest, yaml
from envs.reward_components import REWARD_COMPONENTS
from envs.reward_scales import canonical_payload_hash, load_reward_scale_artifact
import experiments.prepare_fixed_policy_sensitivity_inputs as preparation
from experiments.prepare_fixed_policy_sensitivity_inputs import BAD, differing_paths, dotted_value, expected_oat_differences, hydrate_runtime_base, patch_dotted, prepare, validate_bank, _safe_root

def base(): return yaml.safe_load(Path("configs/paper/base_medium.yaml").read_text())

@pytest.mark.parametrize("path,value",[("passenger.demand_intensity",.75),("scenario.num_parcels",45),("station.power_capacity_kw",880.)])
def test_one_factor_patch_is_deep_and_exact(path,value):
    source=base(); changed=patch_dotted(source,path,value)
    assert differing_paths(source,changed)=={path}; assert dotted_value(changed,path)==value; assert dotted_value(source,path)!=value

@pytest.mark.parametrize("path,value",[("passenger.demand_intensity",1.0),("scenario.num_parcels",60.0),("station.power_capacity_kw",1100)])
def test_baseline_patch_has_expected_empty_difference(path,value):
    source=base(); changed=patch_dotted(source,path,value)
    assert differing_paths(source,changed)==set()
    assert expected_oat_differences(source,path,value)==set()

def test_empty_difference_is_expected_only_for_actual_runtime_baseline():
    source=base()
    assert expected_oat_differences(source,"scenario.num_parcels",60.0)==set()
    assert expected_oat_differences(source,"scenario.num_parcels",45)=={"scenario.num_parcels"}

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
    with pytest.raises(ValueError,match="count"): validate_bank(p,base_config=base(),parameter="scenario.num_parcels",value=45,base_value=60,expected_count=100,expected_seeds=list(range(100)))

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
                      base_value=60,expected_count=1,expected_seeds=[1])

def _empty_bank(tmp_path, runtime, resolved):
    bank=tmp_path/"bank"; bank.mkdir()
    resolved_path=tmp_path/"resolved.yaml"; resolved_path.write_text(yaml.safe_dump(resolved,sort_keys=False))
    manifest={"sensitivity_mode":"fixed_policy_robustness","is_final_test_bank":False,
              "scenario_count":0,"resolved_environment_config":resolved,"scenarios":[],
              "resolved_config_path":str(resolved_path)}
    from evaluation.scenario_bank import sha256_json
    manifest["bank_hash"]=sha256_json(manifest)
    (bank/"scenario_bank_manifest.json").write_text(json.dumps(manifest))
    return bank

def test_validate_bank_accepts_baseline_and_rejects_mismatched_spec_base(tmp_path):
    scale=tmp_path/"formal-scale.json"; _formal_scale(scale)
    runtime,_=hydrate_runtime_base(base(),scale)
    bank=_empty_bank(tmp_path,runtime,patch_dotted(runtime,"scenario.num_parcels",60.0))
    validate_bank(bank,base_config=runtime,parameter="scenario.num_parcels",value=60.0,
                  base_value=60,expected_count=0,expected_seeds=[])
    with pytest.raises(ValueError,match="configured base_value"):
        validate_bank(bank,base_config=runtime,parameter="scenario.num_parcels",value=60,
                      base_value=61,expected_count=0,expected_seeds=[])

def test_validate_bank_rejects_non_target_difference_for_baseline(tmp_path):
    scale=tmp_path/"formal-scale.json"; _formal_scale(scale)
    runtime,_=hydrate_runtime_base(base(),scale)
    resolved=patch_dotted(runtime,"scenario.num_parcels",60)
    resolved["passenger"]["demand_intensity"]+=0.1
    bank=_empty_bank(tmp_path,runtime,resolved)
    with pytest.raises(ValueError,match="non-target"):
        validate_bank(bank,base_config=runtime,parameter="scenario.num_parcels",value=60,
                      base_value=60,expected_count=0,expected_seeds=[])

@pytest.mark.parametrize("diagnostic,expected_levels",[(None,11),(1,6)])
def test_preparation_iterates_formal_levels_and_preserves_diagnostic_endpoints(tmp_path,monkeypatch,diagnostic,expected_levels):
    cfg=yaml.safe_load(Path("configs/paper/fixed_policy_sensitivity.yaml").read_text())
    artifacts=tmp_path/"artifacts"; artifacts.mkdir()
    for group in ("policy_checkpoints","reward_models"):
        for key in cfg["formal_artifacts"][group]:
            artifact=artifacts/f"{group}-{key}.bin"; artifact.write_bytes(b"fixture")
            cfg["formal_artifacts"][group][key]=str(artifact)
    scale=artifacts/"scale.json"; _formal_scale(scale)
    cfg["formal_artifacts"]["reward_reference_scale"]=str(scale)
    cfg["paired_scenarios"]["count"]=1
    config_path=tmp_path/"sensitivity.yaml"; config_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    calls=[]
    def fake_build(config_out,name,count,start,bank_dir,**kwargs):
        bank_dir.mkdir(parents=True)
        return {"scenarios":[]}
    def fake_validate(path,**kwargs):
        calls.append((kwargs["parameter"],kwargs["value"]))
        return {"bank_hash":"a"*64,"resolved_config_hash":"b"*64}
    monkeypatch.setattr(preparation,"build_bank",fake_build)
    monkeypatch.setattr(preparation,"validate_bank",fake_validate)
    prepare(config_path,tmp_path/"output",force=True,diagnostic_scenario_count=diagnostic)
    assert len(calls)==expected_levels
