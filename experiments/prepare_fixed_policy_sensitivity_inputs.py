"""Freeze paired one-factor-at-a-time inputs for fixed-policy robustness."""
from __future__ import annotations
import argparse, copy, json, platform, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import yaml
from data_pipeline.scenario_seeds import derive_scenario_seed_tuple
from evaluation.scenario_bank import load_bank_manifest, load_scenario_bank, sha256_file, sha256_json, verify_scenario_hashes
from experiments.build_scenario_bank import build_bank

MODE = "fixed_policy_robustness"
PROTECTED = tuple(Path(p).resolve() for p in ("results/formal/scenarios", "results/formal/mappo_rlaif_all", "results/formal/reward_models", "results/formal/reward_scales"))
BAD = ("TBD", "REPLACE_WITH_", "PLACEHOLDER", "outputs/reward_reference_scales_v1.json")

def patch_dotted(source: dict[str, Any], dotted: str, value: Any) -> dict[str, Any]:
    out=copy.deepcopy(source); cur=out
    parts=dotted.split(".")
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict): raise KeyError(dotted)
        cur=cur[part]
    if parts[-1] not in cur: raise KeyError(dotted)
    cur[parts[-1]]=value
    return out

def dotted_value(source: dict[str, Any], dotted: str) -> Any:
    cur=source
    for p in dotted.split("."): cur=cur[p]
    return cur

def differing_paths(a: Any, b: Any, prefix="") -> set[str]:
    if isinstance(a,dict) and isinstance(b,dict):
        return set().union(*(differing_paths(a.get(k),b.get(k),f"{prefix}.{k}".strip(".")) for k in a.keys()|b.keys()))
    return set() if a==b else {prefix}

def _safe_root(root: Path) -> Path:
    root=root.resolve()
    if root in PROTECTED or any(p == root or p.is_relative_to(root) for p in PROTECTED):
        raise ValueError(f"refusing protected/non-exclusive output root: {root}")
    expected=(Path("results/formal/sensitivity_fixed").resolve(), Path("results/diagnostic/sensitivity_fixed").resolve())
    if root not in expected and "pytest" not in str(root) and "/tmp/" not in str(root): raise ValueError("output root must be an exclusive sensitivity_fixed directory")
    return root

def _artifacts(cfg):
    f=cfg["formal_artifacts"]
    items={f"policy_{k}":Path(v) for k,v in f["policy_checkpoints"].items()}
    items.update({f"reward_{k}":Path(v) for k,v in f["reward_models"].items()})
    items["reward_scale"]=Path(f["reward_reference_scale"])
    missing=[str(p) for p in items.values() if not p.is_file()]
    if missing: raise FileNotFoundError("missing required formal runtime artifacts:\n"+"\n".join(missing))
    hashes={k:sha256_file(p) for k,p in items.items()}
    if any(any(x in str(p) for x in BAD) for p in items.values()): raise ValueError("placeholder or stale artifact path")
    return items,hashes

def validate_bank(path, *, base_config, parameter, value, expected_count, expected_seeds):
    m=load_bank_manifest(path)
    if m.get("sensitivity_mode")!=MODE or m.get("is_final_test_bank") is not False: raise ValueError("bank is not a fixed-policy sensitivity bank")
    if int(m.get("scenario_count",-1)) != expected_count: raise ValueError("scenario count mismatch")
    if not m.get("bank_hash") or any(x in str(m["bank_hash"]) for x in BAD): raise ValueError("invalid bank hash")
    if dotted_value(m["resolved_environment_config"],parameter)!=value: raise ValueError("target parameter mismatch")
    if differing_paths(base_config,m["resolved_environment_config"]) != {parameter}: raise ValueError("non-target configuration difference")
    scenarios=m.get("scenarios",[])
    if [s.get("paired_scenario_index") for s in scenarios] != list(range(expected_count)): raise ValueError("paired scenario index mismatch")
    if [s.get("base_seed") for s in scenarios] != expected_seeds: raise ValueError("paired seed mismatch")
    for s in load_scenario_bank(path).scenarios: verify_scenario_hashes(s)
    expected=sha256_json({k:v for k,v in m.items() if k not in {"bank_hash","generation_commit"}})
    # enriched manifests intentionally use the canonical complete body hash.
    if m["bank_hash"] != expected: raise ValueError("scenario bank hash mismatch")
    return m

def prepare(config_path, output_root, *, resume=False, force=False, validate_only=False, diagnostic_scenario_count=None):
    if resume and force: raise ValueError("--resume and --force are mutually exclusive")
    cfg=yaml.safe_load(Path(config_path).read_text()); root=_safe_root(Path(output_root)); base_path=Path(cfg["base_environment_config"]); base=yaml.safe_load(base_path.read_text())
    items,hashes=_artifacts(cfg)  # validate-only is deliberately an artifact/lineage check
    classification="diagnostic" if diagnostic_scenario_count is not None else cfg.get("run_classification","formal")
    count=diagnostic_scenario_count or int(cfg["paired_scenarios"]["count"]); start=int(cfg["paired_scenarios"]["seed_start"]); seeds=list(range(start,start+count))
    if force:
        if root.exists(): shutil.rmtree(root)
    records={}
    for family,spec in cfg["families"].items():
        for value,label in zip(spec["values"],spec["labels"]):
            if diagnostic_scenario_count is not None and value not in (spec["values"][0], spec["values"][-1]):
                continue
            resolved=patch_dotted(base,spec["parameter"],value)
            if differing_paths(base,resolved)!={spec["parameter"]}: raise AssertionError("OAT patch failed")
            bank_dir=root/"scenarios"/family/label; config_out=root/"configs"/family/f"{label}.yaml"
            if validate_only:
                if bank_dir.exists(): validate_bank(bank_dir,base_config=base,parameter=spec["parameter"],value=value,expected_count=count,expected_seeds=seeds)
                continue
            if resume:
                if not (bank_dir/"scenario_bank_manifest.json").is_file(): raise RuntimeError(f"stale or partial bank: {bank_dir}")
                m=validate_bank(bank_dir,base_config=base,parameter=spec["parameter"],value=value,expected_count=count,expected_seeds=seeds)
            else:
                config_out.parent.mkdir(parents=True,exist_ok=True); config_out.write_text(yaml.safe_dump(resolved,sort_keys=False))
                m=build_bank(config_out,f"sensitivity_{family}_{label}",count,start,bank_dir,explicit_seeds=",".join(map(str,seeds)),fallback=False,run_classification=classification,force=False)
                for i,s in enumerate(m["scenarios"]): s.update(paired_scenario_index=i,base_seed=seeds[i])
                m.update(sensitivity_mode=MODE,publication_role=MODE,is_final_test_bank=False,publication_eligible=classification=="formal",sensitivity_family=family,parameter_name=spec["parameter"],parameter_value=value,base_parameter_value=spec["base_value"],resolved_environment_config=resolved,resolved_config_path=str(config_out),resolved_config_hash=sha256_file(config_out))
                m["bank_hash"]=sha256_json({k:v for k,v in m.items() if k not in {"bank_hash","generation_commit"}})
                for name in ("scenario_bank_manifest.json","manifest.json"): (bank_dir/name).write_text(json.dumps(m,indent=2,sort_keys=True)+"\n")
                m=validate_bank(bank_dir,base_config=base,parameter=spec["parameter"],value=value,expected_count=count,expected_seeds=seeds)
            records[f"{family}/{label}"]={"path":str(bank_dir/"scenario_bank_manifest.json"),"hash":m["bank_hash"],"resolved_config_path":str(config_out),"resolved_config_hash":m.get("resolved_config_hash")}
    if validate_only: return {"status":"valid","artifacts":hashes}
    try: torch_version=__import__("torch").__version__
    except ImportError: torch_version=None
    manifest={"run_classification":classification,"publication_eligible":classification=="formal","sensitivity_mode":MODE,"publication_role":MODE,"policy_retraining_performed":False,"policy_retraining_statement":"Policy retraining is not performed.","git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"repository_dirty":bool(subprocess.check_output(["git","status","--short"],text=True).strip()),"python_version":platform.python_version(),"pytorch_version":torch_version,"creation_timestamp":datetime.now(timezone.utc).isoformat(),"source_configuration":{"path":str(base_path),"hash":sha256_file(base_path)},"reward_reference_scale":{"path":str(items["reward_scale"]),"hash":hashes["reward_scale"]},"policy_checkpoints":{k.removeprefix("policy_"):{"path":str(items[k]),"hash":v} for k,v in hashes.items() if k.startswith("policy_")},"reward_models":{k.removeprefix("reward_"):{"path":str(items[k]),"hash":v} for k,v in hashes.items() if k.startswith("reward_") and k!="reward_scale"},"paired_scenarios":{"seed_start":start,"count":count,"explicit_values":seeds},"scenario_count_per_level":count,"families":cfg["families"],"scenario_banks":records}
    root.mkdir(parents=True,exist_ok=True); (root/"sensitivity_input_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n"); return manifest

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/paper/fixed_policy_sensitivity.yaml"); p.add_argument("--output-root",default="results/formal/sensitivity_fixed"); p.add_argument("--resume",action="store_true"); p.add_argument("--force",action="store_true"); p.add_argument("--validate-only",action="store_true"); p.add_argument("--diagnostic-scenario-count",type=int)
    a=p.parse_args(argv); print(json.dumps(prepare(a.config,a.output_root,resume=a.resume,force=a.force,validate_only=a.validate_only,diagnostic_scenario_count=a.diagnostic_scenario_count),indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
