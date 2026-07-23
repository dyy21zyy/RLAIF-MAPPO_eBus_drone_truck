"""Estimate reward reference scales from frozen train-bank DynamicDeliveryEnv rollouts."""
from __future__ import annotations
import argparse, csv, json, math, time, subprocess
from pathlib import Path
from typing import Any
import numpy as np, yaml
from envs.delivery_env import DynamicDeliveryEnv
from envs.reward_components import REWARD_COMPONENTS
from envs.reward_scales import canonical_payload_hash, load_reward_scale_artifact
from evaluation.scenario_bank import load_bank_manifest, load_scenario_bank, verify_scenario_hashes, sha256_file, sha256_json
from evaluation.reward_scale_reference_policies import get_reference_policies

class InvalidRewardScaleScenarioBankError(RuntimeError): pass
class RewardScaleEstimationError(RuntimeError): pass

BLOCKING = {"missing", "unexercised"}

def _git():
    try: return subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    except Exception: return "unknown"

def validate_training_bank(path, *, expected_bank_hash=None, formal_mode=False):
    m=load_bank_manifest(path); split=m.get("split", m.get("bank"))
    if split != "train": raise InvalidRewardScaleScenarioBankError(f"reward scales require frozen train bank; got split={split!r}")
    if not m.get("bank_hash"):
        m["bank_hash"] = sha256_json({k:v for k,v in m.items() if k != "bank_hash"})
    if expected_bank_hash and str(expected_bank_hash).startswith("REPLACE_WITH_REAL_"): raise InvalidRewardScaleScenarioBankError("expected bank hash is placeholder")
    if expected_bank_hash and m["bank_hash"] != expected_bank_hash: raise InvalidRewardScaleScenarioBankError("scenario bank hash mismatch")
    if formal_mode and m.get("run_classification") not in {"formal", None}: raise InvalidRewardScaleScenarioBankError("formal scale estimation requires formal training bank")
    bank=load_scenario_bank(path)
    for s in bank.scenarios: verify_scenario_hashes(s)
    return m, bank

def percentile_scale(samples, *, method="percentile", percentile=95, minimum=0.0):
    vals=np.asarray(samples, dtype=float)
    if vals.size == 0: raise RewardScaleEstimationError("no samples")
    if not np.all(np.isfinite(vals)): raise RewardScaleEstimationError("nonfinite sample")
    if np.any(vals < 0): raise RewardScaleEstimationError("negative sample")
    pos=vals[vals>0]
    if pos.size == 0: raise RewardScaleEstimationError("zero scale without override")
    if method == "percentile":
        if not (0 < float(percentile) <= 100): raise RewardScaleEstimationError("percentile must be in (0,100]")
        scale=float(np.percentile(pos, float(percentile)))
    elif method == "median_positive": scale=float(np.median(pos))
    elif method == "mean_positive": scale=float(np.mean(pos))
    elif method == "maximum": scale=float(np.max(pos))
    else: raise RewardScaleEstimationError(f"unknown estimator: {method}")
    scale=max(scale, float(minimum or 0.0))
    if not math.isfinite(scale) or scale <= 0: raise RewardScaleEstimationError("scale must be finite and positive")
    return scale

def classify_and_estimate(rows, cfg):
    est=cfg.get("estimator", {"method":"percentile", "percentile":95})
    overrides=cfg.get("minimum_scale_overrides", {}) or {}
    components={}; scales={}; stats=[]
    valid=[r for r in rows if r.get("episode_status") == "success"]
    for c in REWARD_COMPONENTS:
        key=f"raw_{c}"; vals=[]; missing=0
        for r in valid:
            v=r.get(key)
            if v in (None, ""): missing += 1
            else: vals.append(float(v))
        pos=[v for v in vals if v>0]; zeros=[v for v in vals if v==0]
        if missing: status="missing"
        elif pos: status="observed_positive"
        elif valid: status="instrumented_zero"
        else: status="unexercised"
        override=overrides.get(c)
        selected=None; reason=None; oval=None
        if status == "observed_positive": selected=percentile_scale(vals, method=est.get("method","percentile"), percentile=est.get("percentile",95), minimum=cfg.get("minimum_positive_scale",{}).get(c,0.0) if isinstance(cfg.get("minimum_positive_scale"),dict) else 0.0)
        elif status == "instrumented_zero" and override:
            oval=float(override.get("value")); reason=str(override.get("reason", "")).strip()
            if not math.isfinite(oval) or oval <= 0 or not reason: raise RewardScaleEstimationError(f"invalid minimum override for {c}")
            selected=oval
        elif status == "instrumented_zero":
            selected=None
        arr=np.asarray(vals, dtype=float) if vals else np.asarray([], dtype=float)
        def q(p): return float(np.percentile(arr, p)) if arr.size else None
        rec={"component":c,"status":status,"episode_count":len(rows),"valid_count":len(valid),"missing_count":missing,"zero_count":len(zeros),"positive_count":len(pos),"minimum":float(np.min(arr)) if arr.size else None,"maximum":float(np.max(arr)) if arr.size else None,"mean":float(np.mean(arr)) if arr.size else None,"standard_deviation":float(np.std(arr)) if arr.size else None,"median":q(50),"p75":q(75),"p90":q(90),"p95":q(95),"p99":q(99),"estimator":est.get("method","percentile"),"selected_scale":selected,"minimum_override_value":oval,"minimum_override_reason":reason}
        stats.append(rec); components[c]={"scale":selected,"status":status,"positive_count":len(pos),"minimum_override":({"value":oval,"reason":reason} if oval is not None else None)}
        if selected is not None: scales[c]=selected
    return components, scales, stats

def _select(policy, obs): return policy.select_action(obs)

def run_estimation(scenario_bank, config, output, *, run_classification=None, scenario_limit=None, policies=None, seed=0, force=False, report_only=False, resume=False):
    cfg=yaml.safe_load(Path(config).read_text()) if config else {}
    classification=(run_classification or cfg.get("run_classification") or "diagnostic").lower()
    expected=cfg.get("scenario_bank",{}).get("expected_bank_hash")
    manifest, bank=validate_training_bank(scenario_bank, expected_bank_hash=expected, formal_mode=(classification=="formal"))
    if report_only:
        print(json.dumps({"report_only":True,"training_scenario_bank_hash":manifest["bank_hash"],"scenario_count":len(bank.scenarios),"run_classification":classification}, indent=2)); return None
    out=Path(output); outdir=out.parent
    if out.exists() and not force: raise FileExistsError(f"output exists; use --force: {out}")
    outdir.mkdir(parents=True, exist_ok=True)
    resolved_cfg=json.loads(json.dumps(cfg, sort_keys=True, default=str)); resolved_cfg["run_classification"]=classification; resolved_cfg["scenario_bank"]={**resolved_cfg.get("scenario_bank",{}),"manifest":str(scenario_bank),"bank_hash":manifest["bank_hash"]}
    resolved_hash=sha256_json(resolved_cfg)
    selected=bank.scenarios[:scenario_limit] if scenario_limit else bank.scenarios
    pols=get_reference_policies(policies or cfg.get("reference_policies"))
    progress_path = outdir / "reward_scale_episode_components.progress.jsonl"
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if resume and progress_path.exists() and not force:
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                completed[(str(rec.get("reference_policy")), str(rec.get("scenario_id")))] = rec
    rows=[]; failures=[]; steps_total=0
    progress_handle = progress_path.open("a", encoding="utf-8")
    try:
        for pol in pols:
            for s in selected:
                key = (pol.name, s.scenario_id)
                if key in completed:
                    row = completed[key]
                    rows.append(row)
                    if row.get("episode_status") == "failed": failures.append(row)
                    steps_total += int(row.get("transition_count") or 0)
                    continue
                t=time.time(); base={"scenario_id":s.scenario_id,"scenario_content_hash":s.scenario_content_hash,"instance_hash":s.instance_hash,"scenario_bank_hash":bank.bank_hash,"reference_policy":pol.name,"reference_policy_version":pol.version,"estimation_seed":seed,"run_classification":classification}
                try:
                    env=DynamicDeliveryEnv(s.instance_path)
                    env.config.setdefault("reward", {})["apply_reference_scales"] = False
                    env.reward_reference_scales={c:1.0 for c in REWARD_COMPONENTS}
                    obs,_=env.reset(seed=seed)
                    transitions=0; term=trunc=False; info={}
                    while obs.get("agent_id") != "terminal" and not (term or trunc):
                        obs, reward, term, trunc, info = env.step(_select(pol, obs)); transitions += 1
                        if transitions > int(cfg.get("maximum_transitions", 10000)): raise RuntimeError("maximum transitions exceeded")
                    raw=dict(getattr(env,"raw_cost_components", {}) or info.get("raw_cost_components", {}))
                    row={**base,"episode_status":"success","transition_count":transitions,"runtime":time.time()-t,"released_parcels":len(getattr(env,"parcels",{})),"delivered_parcels":(info.get("delivered_parcels", 0) if isinstance(info.get("delivered_parcels", 0), int) else len(info.get("delivered_parcels", []))) if isinstance(info,dict) else 0,"failure_reason":"","exception_type":""}
                    for c in REWARD_COMPONENTS: row[f"raw_{c}"]=raw.get(c, None)
                    if transitions <= 0: raise RuntimeError("no real env.step occurred")
                    rows.append(row); steps_total += transitions
                    progress_handle.write(json.dumps(row, sort_keys=True)+"\n"); progress_handle.flush()
                except Exception as exc:
                    row={**base,"episode_status":"failed","transition_count":0,"runtime":time.time()-t,"released_parcels":None,"delivered_parcels":None,"failure_reason":str(exc),"exception_type":exc.__class__.__name__}
                    for c in REWARD_COMPONENTS: row[f"raw_{c}"]=None
                    rows.append(row); failures.append(row)
                    progress_handle.write(json.dumps(row, sort_keys=True)+"\n"); progress_handle.flush()
    finally:
        progress_handle.close()
    rows.sort(key=lambda r: (str(r.get("reference_policy")), str(r.get("scenario_id"))))
    failures.sort(key=lambda r: (str(r.get("reference_policy")), str(r.get("scenario_id"))))
    ep_csv=outdir/"reward_scale_episode_components.csv"; ep_jsonl=outdir/"reward_scale_episode_components.jsonl"
    fields=list(rows[0].keys()) if rows else []
    with ep_csv.open("w", newline="") as f: w=csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)
    ep_jsonl.write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in rows))
    components, scales, stats=classify_and_estimate(rows, cfg)
    failure_fraction=len(failures)/max(len(rows),1)
    min_valid=int(cfg.get("minimum_valid_episodes",1)); max_fail=float(cfg.get("maximum_failure_fraction", 1.0 if classification=="diagnostic" else 0.0))
    valid=sum(1 for r in rows if r["episode_status"]=="success")
    blocked=[c for c,m in components.items() if m["status"] in BLOCKING or m["scale"] is None]
    passed=steps_total>0 and valid>=min_valid and failure_fraction<=max_fail and not blocked
    if classification=="formal" and not passed: raise RewardScaleEstimationError(f"formal reward-scale validation blocked: {blocked}")
    stat_csv=outdir/"reward_scale_statistics.csv"; stat_json=outdir/"reward_scale_statistics.json"
    with stat_csv.open("w", newline="") as f: w=csv.DictWriter(f, stats[0].keys()); w.writeheader(); w.writerows(stats)
    stat_json.write_text(json.dumps(stats, indent=2, sort_keys=True))
    (outdir/"reward_scale_failures.json").write_text(json.dumps(failures, indent=2, sort_keys=True))
    artifact={"artifact_type":"reward_reference_scales","artifact_version":1,"run_classification":classification,"validation_status":"passed" if passed else "blocked","component_order":list(REWARD_COMPONENTS),"training_scenario_bank_path":str(scenario_bank),"training_scenario_bank_hash":bank.bank_hash,"training_scenario_count":len(bank.scenarios),"reference_policy_suite":[p.metadata() for p in pols],"estimator":cfg.get("estimator", {"method":"percentile","percentile":95}),"components":components,"scales":scales,"source_episode_file_hash":sha256_file(ep_jsonl),"statistics_file_hash":sha256_file(stat_json),"resolved_config_hash":resolved_hash,"code_commit":_git(),"creation_timestamp":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    artifact["artifact_hash"]=canonical_payload_hash(artifact)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True)+"\n")
    (outdir/"reward_scale_manifest.json").write_text(json.dumps({"artifact":out.name,"artifact_hash":artifact["artifact_hash"],"validation_status":artifact["validation_status"],"runtime_files":[ep_csv.name,ep_jsonl.name,stat_csv.name,stat_json.name,"reward_scale_failures.json"]}, indent=2, sort_keys=True))
    load_reward_scale_artifact(out, expected_hash=artifact["artifact_hash"], expected_training_bank_hash=bank.bank_hash, formal_mode=False)
    return artifact

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--scenario-bank", required=True); ap.add_argument("--config", required=True); ap.add_argument("--output", required=True); ap.add_argument("--run-classification"); ap.add_argument("--scenario-limit", type=int); ap.add_argument("--policy", action="append"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--resume", action="store_true"); ap.add_argument("--workers", type=int, default=1, help="Reserved bounded worker count; deterministic serial execution is used when 1."); ap.add_argument("--report-only", action="store_true"); ap.add_argument("--force", action="store_true")
    a=ap.parse_args(); r=run_estimation(a.scenario_bank,a.config,a.output,run_classification=a.run_classification,scenario_limit=a.scenario_limit,policies=a.policy,seed=a.seed,force=a.force,report_only=a.report_only,resume=a.resume)
    if r is not None: print(json.dumps({"artifact":a.output,"artifact_hash":r["artifact_hash"],"validation_status":r["validation_status"],"scales":r["scales"],"statuses":{k:v["status"] for k,v in r["components"].items()}}, indent=2, sort_keys=True))
if __name__ == "__main__": main()
