"""Deterministically evaluate immutable four-agent MAPPO policies on sensitivity banks."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, platform, subprocess, sys, time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import yaml
from envs import DynamicDeliveryEnv
import evaluation.metrics
from evaluation.scenario_bank import load_scenario_bank, load_frozen_instance, sha256_file, verify_scenario_hashes
from rlaif.reward_registry import RewardRegistry
from training.event_schema import decision_event_id, normalize_decision_event_type, validate_agent_event
from training.mappo_trainer import _candidate_feature_payload, _pad_vector, load_checkpoint

AGENTS=("assignment","truck","bus","station")
IDENTITY_FIELDS=("sensitivity_mode","sensitivity_family","parameter_name","parameter_value","policy_seed","policy_checkpoint_hash","scenario_bank_hash","scenario_id","scenario_content_hash")

def evaluation_identity(row): return tuple(str(row.get(k)) for k in IDENTITY_FIELDS)
def should_skip(rows, identity): return any(r.get("status")=="success" and evaluation_identity(r)==identity for r in rows)
def reject_duplicate_identities(rows):
    ids=[evaluation_identity(r) for r in rows]
    if len(ids)!=len(set(ids)): raise ValueError("duplicated evaluation identity")

def _hash_parameters(module):
    h=hashlib.sha256()
    for p in module.parameters(): h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def _registry(cfg, manifest):
    agents={a:{"enabled":True,"checkpoint":manifest["reward_models"][a]["path"],"checkpoint_hash":manifest["reward_models"][a]["hash"]} for a in AGENTS}
    return RewardRegistry({"run_classification":manifest["run_classification"],"rlaif":{"enabled":True,"scope":"all","fallback_to_env_reward":False,"fail_on_invalid_reward_model":True,"agents":agents}})

def _initial_audit():
    return ({f"rlaif_reward_{a}":0.0 for a in AGENTS}
            | {f"{a}_decision_count":0 for a in AGENTS}
            | {"fallback_count":0,"total_reward_clipping_count":0,
               "selected_station_dispatches":0,"station_dispatch_action_count":0,
               "station_decisions_with_dispatch_option":0,
               "station_decisions_without_dispatch_option":0,
               "assignment_td_count":0,"assignment_tbd_count":0,
               "assignment_tld_count":0})

def _audit_selected_action(obs, action, audit):
    """Audit explicit candidate semantics before the environment mutates state."""
    candidates=obs["candidate_actions"]
    selected=candidates[action]
    agent=str(obs["agent_id"])
    selected_mode=None
    selected_dispatch_count=None
    if agent=="station":
        features=selected["features"]
        selected_dispatch_count=int(features["dispatch_count"])
        payload=features["dispatch_payload"]
        if selected_dispatch_count != len(payload):
            raise RuntimeError("selected station dispatch_count differs from dispatch_payload length")
        audit["selected_station_dispatches"]+=selected_dispatch_count
        audit["station_dispatch_action_count"]+=int(selected_dispatch_count>0)
        has_option=any(
            bool(candidate["feasible"])
            and int(candidate["features"]["dispatch_count"])>0
            for candidate in candidates
        )
        audit["station_decisions_with_dispatch_option"]+=int(has_option)
        audit["station_decisions_without_dispatch_option"]+=int(not has_option)
    elif agent=="assignment":
        features=selected["features"]
        modes=[]
        for mode,key in (("TD","action_type_TD"),("TBD","action_type_TBD"),("TLD","action_type_TLD")):
            if int(features[key]) != 0:
                audit[f"assignment_{mode.lower()}_count"]+=1
                modes.append(mode)
        selected_mode=modes[0] if len(modes)==1 else None
    return {
        "agent":agent,
        "event_type":str(obs["event_type"]),
        "selected_action_index":int(action),
        "selected_action_type":str(selected["action_type"]),
        "entity_id":str(selected["entity_id"]),
        "selected_station_dispatch_count":selected_dispatch_count,
        "selected_assignment_mode":selected_mode,
    }

def _validate_episode_audit(metrics, audit):
    if int(metrics["drone_missions"]) != audit["selected_station_dispatches"]:
        raise RuntimeError("drone_mission_count differs from selected station dispatches")
    assignment_modes=sum(audit[f"assignment_{mode}_count"] for mode in ("td","tbd","tld"))
    if assignment_modes != audit["assignment_decision_count"]:
        raise RuntimeError("assignment mode counts differ from assignment decision count")
    audit["drone_mission_counter_consistent"]=1

def _flat_metrics(env, audit):
    # This strict collector is the single source of physical publication metrics.
    # FormalMetricError deliberately propagates so a row cannot be marked successful
    # after missing instrumentation has been replaced by a fabricated zero.
    canonical=asdict(evaluation.metrics.collect_formal_runtime_metrics(env))
    released=sum(getattr(p,"release_time_min",None) is not None for p in env.parcels.values())
    delivered=released-canonical["undelivered_parcels"]
    boarded=env.passenger_boardings_at_ordinary_stops
    capacity=float(env.config["station"]["power_capacity_kw"])
    if capacity <= 0: raise ValueError("configured station power capacity must be positive")
    _validate_episode_audit(canonical,audit)
    out={**canonical,"released_parcels":released,"delivered_parcels":delivered,"undelivered_rate":canonical["undelivered_parcels"]/released if released else 0.0,"truck_distance_per_released_parcel":canonical["truck_distance"]/released if released else 0.0,"drone_missions_per_released_parcel":canonical["drone_missions"]/released if released else 0.0,"total_boarded_passengers":boarded,"waiting_minutes_per_passenger":canonical["waiting_passenger_minutes"]/boarded if boarded else 0.0,"onboard_delay_minutes_per_passenger":canonical["onboard_additional_delay_passenger_minutes"]/boarded if boarded else 0.0,"configured_station_power_capacity_kw":capacity,"peak_load_to_capacity_ratio":canonical["station_peak_power"]/capacity,"overload_episode_indicator":int(canonical["overload_kw_min"]>0),**audit}
    for agent in AGENTS:
        count=audit[f"{agent}_decision_count"]
        out[f"{agent}_reward_per_decision"]=audit[f"rlaif_reward_{agent}"]/count if count else None
    if any(isinstance(v,float) and not math.isfinite(v) for v in out.values()): raise ValueError("non-finite required metric")
    return out

def evaluate_scenario(scenario, actors, registry, *, policy_seed, configured_parameter, configured_value):
    verify_scenario_hashes(scenario); instance=load_frozen_instance(scenario); snap=instance.get("config_snapshot",{})
    cur=snap
    for part in configured_parameter.split("."): cur=cur[part]
    if float(cur)!=float(configured_value): raise ValueError("inconsistent configured sensitivity parameter")
    env=DynamicDeliveryEnv(Path(scenario.instance_path)); obs,_=env.reset(seed=policy_seed); before=_hash_parameters(actors); trace=[]; audit=_initial_audit()
    terminal=False
    with __import__("torch").inference_mode():
        for _ in range(10000):
            if obs.get("agent_id")=="terminal": terminal=True; break
            agent=str(obs["agent_id"]); event=normalize_decision_event_type(obs["event_type"]); validate_agent_event(agent,event)
            raw=[float(x) for x in obs["features"]]; candidates,names=_candidate_feature_payload(obs); mask=[bool(x) for x in obs["action_mask"]]; actor=actors[agent]
            action,_=actor.act(_pad_vector(raw,actor.obs_dim),decision_event_id(event),candidates,mask,deterministic=True)
            if not mask[action]: raise RuntimeError("deterministic masked action infeasible")
            trace.append(_audit_selected_action(obs,action,audit))
            nxt,reward,terminated,truncated,_=env.step(action)
            contribution=registry.score_transition(agent_type=agent,event_type=event,environment_reward=float(reward),state_features=raw,candidate_features=candidates[action],selected_action_index=action,formal_mode=True)
            audit[f"rlaif_reward_{agent}"]+=float(contribution.weighted_learned_contribution); audit[f"{agent}_decision_count"]+=1; audit["fallback_count"]+=int(contribution.used_fallback); obs=nxt
            if terminated or truncated: terminal=bool(terminated); break
    if not terminal: raise RuntimeError("non-terminal or incomplete episode")
    if audit["fallback_count"]: raise RuntimeError("reward-model fallback")
    if before!=_hash_parameters(actors): raise RuntimeError("policy parameters mutated during evaluation")
    return _flat_metrics(env,audit),trace

def run(config_path, output_root, *, family=None, value=None, policy_seed=None, scenario_limit=None, resume=False, validate_only=False, continue_on_error=False):
    cfg=yaml.safe_load(Path(config_path).read_text()); root=Path(output_root); im=json.loads((root/"sensitivity_input_manifest.json").read_text())
    if cfg.get("sensitivity_mode")!="fixed_policy_robustness" or cfg["evaluation"].get("fallback_to_env_reward") is not False or cfg["evaluation"].get("deterministic") is not True: raise ValueError("invalid fixed-policy evaluation configuration")
    scale=im["reward_reference_scale"];
    if sha256_file(scale["path"])!=scale["hash"]: raise ValueError("reward-scale hash mismatch")
    selected=[]
    for fam,spec in cfg["families"].items():
        if family and fam!=family: continue
        for val,label in zip(spec["values"],spec["labels"]):
            if value is not None and float(val)!=float(value): continue
            if f"{fam}/{label}" not in im["scenario_banks"]: continue
            selected.append((fam,spec,val,label))
    seeds=[policy_seed] if policy_seed else [1,2,3]
    if validate_only:
        for seed in seeds:
            rec=im["policy_checkpoints"][str(seed)]
            if sha256_file(rec["path"])!=rec["hash"]: raise ValueError("policy checkpoint hash mismatch")
        for rec in im["reward_models"].values():
            if sha256_file(rec["path"])!=rec["hash"]: raise ValueError("reward-model hash mismatch")
        return []
    rows=[]
    for fam,spec,val,label in selected:
      bankrec=im["scenario_banks"][f"{fam}/{label}"]; bank=load_scenario_bank(bankrec["path"])
      if bank.bank_hash!=bankrec["hash"]: raise ValueError("scenario-bank hash mismatch")
      for seed in seeds:
        ck=im["policy_checkpoints"][str(seed)]
        if sha256_file(ck["path"])!=ck["hash"]: raise ValueError("policy checkpoint hash mismatch")
        actors,critic,_=load_checkpoint(ck["path"]); actors.eval(); critic.eval(); registry=_registry(cfg,im)
        out=root/"evaluation"/fam/label/f"seed_{seed}"; csvpath=out/"episodes.csv"; existing=list(csv.DictReader(csvpath.open())) if resume and csvpath.is_file() else []
        reject_duplicate_identities(existing)
        started=time.time(); current=[]
        for i,sc in enumerate(bank.scenarios[:scenario_limit]):
            base={"run_classification":im["run_classification"],"publication_eligible":bool(im["publication_eligible"]),"sensitivity_mode":"fixed_policy_robustness","publication_role":"fixed_policy_robustness","sensitivity_family":fam,"parameter_name":spec["parameter"],"parameter_value":val,"base_parameter_value":spec["base_value"],"policy_seed":seed,"policy_checkpoint":ck["path"],"policy_checkpoint_hash":ck["hash"],"scenario_id":sc.scenario_id,"paired_scenario_index":i,"scenario_bank_hash":bank.bank_hash,"instance_hash":sc.instance_hash,"scenario_content_hash":sc.scenario_content_hash,"reward_scale_path":scale["path"],"reward_scale_hash":scale["hash"],**{f"reward_{a}_hash":im["reward_models"][a]["hash"] for a in AGENTS}}
            ident=evaluation_identity(base)
            if resume and should_skip(existing,ident): continue
            t=time.perf_counter()
            try: metrics,trace=evaluate_scenario(sc,actors,registry,policy_seed=seed,configured_parameter=spec["parameter"],configured_value=val); base.update(metrics,status="success",error_message="",action_trace_hash=hashlib.sha256(json.dumps(trace,sort_keys=True,separators=(",",":")).encode()).hexdigest())
            except Exception as exc:
                base.update(status="failed",error_message=str(exc))
                if not continue_on_error: raise
            base["runtime_seconds"]=time.perf_counter()-t; current.append(base)
        allrows=existing+current; reject_duplicate_identities(allrows); out.mkdir(parents=True,exist_ok=True)
        fields=sorted(set().union(*(r.keys() for r in allrows)))
        with csvpath.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(allrows)
        manifest={"exact_command":" ".join(sys.argv),"start_timestamp":datetime.fromtimestamp(started,timezone.utc).isoformat(),"completion_timestamp":datetime.now(timezone.utc).isoformat(),"runtime_seconds":time.time()-started,"hardware":platform.platform(),"python_version":platform.python_version(),"pytorch_version":__import__("torch").__version__,"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"repository_dirty":bool(subprocess.check_output(["git","status","--short"],text=True).strip()),"policy_path":ck["path"],"policy_hash":ck["hash"],"reward_models":im["reward_models"],"reward_reference_scale":scale,"scenario_bank_path":bankrec["path"],"scenario_bank_hash":bank.bank_hash,"sensitivity_family":fam,"parameter_name":spec["parameter"],"parameter_value":val,"successful_scenario_count":sum(r["status"]=="success" for r in allrows),"failure_count":sum(r["status"]!="success" for r in allrows),"fallback_count":sum(int(r.get("fallback_count",0)) for r in allrows),"status":"success" if all(r["status"]=="success" for r in allrows) else "failed"}
        (out/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n"); rows.extend(current)
    return rows

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/paper/fixed_policy_sensitivity.yaml"); p.add_argument("--output-root",default="results/formal/sensitivity_fixed"); p.add_argument("--resume",action="store_true"); p.add_argument("--validate-only",action="store_true"); p.add_argument("--family",choices=["passenger","parcel","power"]); p.add_argument("--value",type=float); p.add_argument("--policy-seed",type=int,choices=[1,2,3]); p.add_argument("--scenario-limit",type=int); p.add_argument("--continue-on-error",action="store_true")
    a=p.parse_args(argv); rows=run(a.config,a.output_root,family=a.family,value=a.value,policy_seed=a.policy_seed,scenario_limit=a.scenario_limit,resume=a.resume,validate_only=a.validate_only,continue_on_error=a.continue_on_error); print(json.dumps({"episodes":len(rows)})); return 0
if __name__=="__main__": raise SystemExit(main())
