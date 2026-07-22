"""Generate runtime-only diagnostic benchmark fixtures."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import yaml, torch
from data_pipeline.build_instance import build_instance
from evaluation.scenario_bank import freeze_scenario, write_bank_manifest, sha256_file, sha256_json, git_commit
from evaluation.formal_policy_registry import EXPECTED_ALGORITHMS, EXPECTED_SCOPE, FORMAL_METHOD_REGISTRY
from training.event_schema import EVENT_NAME_TO_ID, EVENT_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION, CANDIDATE_SCHEMA_VERSION

CONFIG = Path(__file__).parents[1] / "configs/shanghai_small.yaml"

def _ck(path: Path, method: str, seed: int, reward_hashes=None):
    spec=FORMAL_METHOD_REGISTRY[method]
    data={"checkpoint_schema_version":1,"algorithm":EXPECTED_ALGORITHMS[method],"rlaif_scope":EXPECTED_SCOPE[method],"enabled_reward_agents":list(spec.enabled_reward_agents),"training_seed":seed,"run_classification":"diagnostic","validation_status":"diagnostic_only","training_scenario_bank_hash":"diagnostic","resolved_training_config_hash":"diagnostic","code_commit":git_commit(),"event_name_to_id":dict(EVENT_NAME_TO_ID),"event_schema_version":EVENT_SCHEMA_VERSION,"observation_schema_version":OBSERVATION_SCHEMA_VERSION,"candidate_schema_version":CANDIDATE_SCHEMA_VERSION,"actor_specs":{"assignment":{},"truck":{},"bus":{},"station":{}},"reward_checkpoint_hashes":reward_hashes or {}}
    path.parent.mkdir(parents=True, exist_ok=True); torch.save(data, path); return path

def _reward(path: Path, agent: str):
    data={"checkpoint_type":"agent_reward_model","checkpoint_schema_version":1,"run_classification":"diagnostic","validation_status":"diagnostic_only","agent_type":agent,"event_name_to_id":dict(EVENT_NAME_TO_ID),"constant_reward":0.1}
    path.parent.mkdir(parents=True, exist_ok=True); torch.save(data,path); return path

def build_fixture(output_root: Path, *, force: bool=False, scenario_count: int=3, seed: int=1) -> Path:
    if output_root.exists():
        if not force: raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    scen_root=output_root/"scenarios"/"test"; scenarios=[]
    for i in range(scenario_count):
        tmp=output_root/"_build"/f"src_{i}"
        inst=build_instance(CONFIG, fallback=True, output_root=tmp)
        # Keep diagnostic scenarios content-distinct while using the production instance builder.
        ip=Path(inst["output_directory"])/"instance.json"
        import json as _json
        data=_json.loads(ip.read_text()); data["diagnostic_variant_index"]=i; ip.write_text(_json.dumps(data, indent=2, sort_keys=True)+"\n")
        parcels=Path(inst["output_directory"])/data["artifacts"].get("parcels","parcels.csv")
        if parcels.exists():
            txt=parcels.read_text().splitlines()
            if len(txt)>1:
                cols=txt[0].split(","); vals=txt[1].split(",")
                if "release_time_min" in cols:
                    j=cols.index("release_time_min"); vals[j]=str(float(vals[j]) + i * 0.001); txt[1]=",".join(vals); parcels.write_text("\n".join(txt)+"\n")
        sid=f"test_{i:04d}"
        scenarios.append(freeze_scenario(Path(inst["output_directory"])/"instance.json", scen_root/sid, sid, {"fixture_seed":seed+i}, {"config_path":str(CONFIG)}, split="test", run_classification="diagnostic", fallback=True))
    manifest=write_bank_manifest(scen_root,"test",scenarios,{"config_path":str(CONFIG)},run_classification="diagnostic")
    rewards={a:_reward(output_root/"reward_models"/f"reward_{a}.pt",a) for a in ("assignment","truck","bus","station")}
    rh={a:sha256_file(p) for a,p in rewards.items()}
    policies={
        "assignment_ppo": _ck(output_root/"policies"/"assignment_ppo_seed_1.pt","assignment_ppo",seed),
        "mappo_env": _ck(output_root/"policies"/"mappo_env_seed_1.pt","mappo_env",seed),
        "mappo_rlaif_assignment": _ck(output_root/"policies"/"mappo_rlaif_assignment_seed_1.pt","mappo_rlaif_assignment",seed,{"assignment":rh["assignment"]}),
        "mappo_rlaif_all": _ck(output_root/"policies"/"mappo_rlaif_all_seed_1.pt","mappo_rlaif_all",seed,rh),
    }
    cfg={
        "run_classification":"diagnostic",
        "scenario_bank":{"manifest":str(scen_root/"scenario_bank_manifest.json"),"split":"test","expected_count":scenario_count,"expected_bank_hash":manifest["bank_hash"]},
        "paired_evaluation":True,"fallback":True,"fail_on_missing_artifact":True,"fail_on_missing_metric":True,"training_seeds":[seed],"output_root":str(output_root.parent/"benchmark_real"),
        "methods":[
            {"method_id":"truck_direct_heuristic"},
            {"method_id":"integrated_rule_based"},
            {"method_id":"mappo_env","policy_checkpoints":{str(seed):str(policies["mappo_env"])}},
            {"method_id":"mappo_rlaif_assignment","policy_checkpoints":{str(seed):str(policies["mappo_rlaif_assignment"])},"reward_checkpoints":{"assignment":str(rewards["assignment"])}},
            {"method_id":"mappo_rlaif_all","policy_checkpoints":{str(seed):str(policies["mappo_rlaif_all"])},"reward_checkpoints":{a:str(p) for a,p in rewards.items()}},
        ],
    }
    out=output_root/"benchmark_real.resolved.yaml"; out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output_root/"fixture_manifest.json").write_text(json.dumps({"config_hash":sha256_json(cfg),"scenario_count":scenario_count,"policy_paths":{k:str(v) for k,v in policies.items()}},indent=2))
    shutil.rmtree(output_root/"_build", ignore_errors=True)
    return out

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--output-root',required=True); ap.add_argument('--force',action='store_true'); ap.add_argument('--scenario-count',type=int,default=3); ap.add_argument('--seed',type=int,default=1)
    a=ap.parse_args(argv); path=build_fixture(Path(a.output_root), force=a.force, scenario_count=a.scenario_count, seed=a.seed); print(path); return 0
if __name__=='__main__': raise SystemExit(main())
