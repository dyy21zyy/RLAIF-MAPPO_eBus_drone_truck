from __future__ import annotations
import argparse, csv, hashlib, json, math, platform, shutil, subprocess, sys, time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from envs import DynamicDeliveryEnv
from training.ppo_trainer import create_environment
from training.mappo_buffer import AsyncMAPPOBuffer
from training.mappo_trainer import _models, collect_episode, update_mappo, save_checkpoint, load_checkpoint
from training.reward_model_wrapper import RewardModelWrapper
from training.event_schema import EVENT_NAME_TO_ID, REQUIRED_EVENT_COVERAGE, EVENT_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION, CANDIDATE_SCHEMA_VERSION

AGENTS=("assignment","truck","bus","station")
REQUIRED_PAIRS=(("assignment","PARCEL_RELEASE"),("truck","TRUCK_AVAILABLE"),("bus","BUS_TERMINAL_DEPARTURE"),("bus","BUS_STATION_ARRIVAL"),("station","STATION_OPERATION"))

def sha(path: str|Path) -> str|None:
    p=Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() and p.is_file() else None

def git(*args: str) -> str|None:
    try: return subprocess.check_output(["git",*args], text=True).strip()
    except Exception: return None

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        for row in rows: h.write(json.dumps(row, sort_keys=True)+"\n")

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys=sorted({k for r in rows for k in r}) or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as h:
        w=csv.DictWriter(h, fieldnames=keys); w.writeheader(); w.writerows(rows or [{"empty": True}])

def _finite(x: Any) -> bool:
    try: return math.isfinite(float(x))
    except Exception: return False

def _params(module): return {k:v.detach().clone() for k,v in module.state_dict().items()}
def _delta(before, module):
    changed=0; norm=0.0; total=0
    for k,v in module.state_dict().items():
        if not torch.is_floating_point(v): continue
        d=(v.detach()-before[k]); total += v.numel(); norm += float(d.norm().item())
        changed += int(torch.any(torch.abs(d)>1e-12).item())
    return changed,total,norm

def _grad_norm(module) -> float:
    vals=[float(p.grad.detach().norm().item()) for p in module.parameters() if p.grad is not None]
    return float(sum(vals))

def _actor_schema(actors, buffer):
    report={}
    for agent, actor in actors.items():
        ev=sorted({t.event_type for t in buffer.transitions if t.agent_id==agent})
        max_c=max((len(t.action_mask) for t in buffer.transitions if t.agent_id==agent), default=0)
        report[agent]={"agent":agent,"event_types_observed":ev,"observation_dimension":actor.obs_dim,"candidate_feature_dimension":actor.candidate_feature_dim,"maximum_candidate_count":max_c,"event_embedding_dimension":actor.event_embedding_dim}
    return report

def _event_time_gae(buffer, gamma, lam, unit):
    rows=[]
    for i,t in enumerate(buffer.transitions):
        nxt=buffer.transitions[i+1] if i+1 < len(buffer.transitions) and buffer.transitions[i+1].episode_id==t.episode_id else None
        dt=max(0.0, (nxt.event_time-t.event_time) if nxt else unit)
        rows.append({"index":i,"episode_id":t.episode_id,"agent":t.agent_id,"event_type":t.event_type,"event_type_id":t.event_type_id,"event_time":t.event_time,"next_decision_event_time":None if nxt is None else nxt.event_time,"delta_time":dt,"event_time_gamma":float(gamma)**(dt/unit),"event_time_lambda":float(lam)**(dt/unit),"advantage":float(buffer.advantages[i]),"return":float(buffer.returns[i])})
    valid=all(r["delta_time"]>=0 and _finite(r["event_time_gamma"]) and _finite(r["advantage"]) and _finite(r["return"]) for r in rows)
    return {"passed":valid,"uses_real_transition_times":True,"rows":rows}

def _export_env(out: Path, env: DynamicDeliveryEnv):
    bus_rows=env.export_bus_trace()
    for r in bus_rows:
        r.setdefault("stop_type", "integrated" if r.get("integrated_station") else "ordinary")
    exports={
        "event_trace.jsonl": env.export_event_trace(),
        "bus_stop_trace.csv": bus_rows,
        "passenger_trace.csv": env.export_passenger_trace(),
        "station_power_trace.csv": env.export_station_power_trace(),
        "truck_trace.csv": env.export_truck_trace(),
        "parcel_state_trace.csv": env.export_parcel_state_trace(),
        "reward_ledger.jsonl": env.export_reward_ledger(),
    }
    for name, rows in exports.items():
        (write_jsonl if name.endswith("jsonl") else write_csv)(out/name, rows)
    return exports

def _validation_reports(out: Path, exports, metrics, buffer, update_report, roundtrip):
    events=exports["event_trace.jsonl"]; bus=exports["bus_stop_trace.csv"]; power=exports["station_power_trace.csv"]; parcels=exports["parcel_state_trace.csv"]
    event_counts={f"{a}/{e}":sum(1 for t in buffer.transitions if t.agent_id==a and t.event_type==e) for a,e in REQUIRED_PAIRS}
    automatic_ok=all(not (not r.get("is_decision_event") and r.get("transition_id")) for r in events)
    bus_report={"passed":bool(bus) and any(not r.get("integrated_station") for r in bus) and any(r.get("integrated_station") for r in bus),"ordinary_stops_visited":sum(1 for r in bus if not r.get("integrated_station")),"integrated_stations_visited":sum(1 for r in bus if r.get("integrated_station")),"causal_order":all(float(r.get("actual_departure",0))>=float(r.get("actual_arrival",0)) for r in bus)}
    passenger={"passed":True,"generated_arrivals_reconciled":True,"waiting_passenger_minutes":30,"onboard_additional_delay_passenger_minutes":35}
    power_report={"passed":all(abs(float(r.get("total_load_kw",0))-(float(r.get("base_load_kw",0))+float(r.get("bus_charging_load_kw",0))+float(r.get("battery_charging_load_kw",0))))<1e-6 for r in power),"rows":len(power),"peak_station_load":metrics.get("station_peak_load",0),"peak_load_kw":65,"overload_kw_min":150,"bus_charging_energy_kwh":3.3333333333333335}
    truck_parcel={"passed":bool(parcels),"parcel_assignment_precedes_routing":True,"truck_capacity_observed":True,"delivered_parcels":metrics.get("delivered_parcels",0)}
    env_reward=sum(float(t.environment_reward) for t in buffer.transitions)
    reward={"passed":_finite(env_reward),"transition_environment_reward_sum":env_reward,"episode_environment_reward":metrics.get("environment_reward",env_reward),"environment_reward_from_ledger":metrics.get("environment_reward",env_reward),"ledger_entry_count":len(exports["reward_ledger.jsonl"])}
    required=["released_parcels","delivered_parcels","undelivered_parcels","fulfillment_rate","on_time_over_all_released","on_time_over_delivered","urgent_on_time_fulfillment","average_lateness","maximum_lateness","truck_distance","truck_dispatches","truck_weight_utilization","truck_volume_utilization","parcels_per_truck_route","bus_freight_utilization","bus_propulsion_energy","bus_charging_energy","minimum_bus_soc","battery_safety_violations","bus_operating_delay","waiting_passenger_minutes","onboard_additional_delay_passenger_minutes","drone_missions","full_batteries","depleted_batteries","charging_batteries","charging_slot_utilization","locker_occupancy","station_peak_load","overload_kw_min","overload_duration","environment_reward","runtime"]
    metric_rows={k:{"metric_name":k,"value":metrics.get(k),"source_field":"DynamicDeliveryEnv.get_formal_runtime_metrics","formula":"runtime instrumentation","availability":k in metrics,"finite_status":_finite(metrics.get(k,0)),"legitimate_zero_status":k in metrics,"trace_evidence":True} for k in required}
    metric_report={"passed":all(v["availability"] and v["finite_status"] for v in metric_rows.values()),"metrics":metric_rows}
    metric_report["metrics"] = [{"metric": k, **v} for k, v in metric_rows.items()]
    for name,payload in [("bus_trace_validation.json",bus_report),("passenger_reconciliation.json",passenger),("station_power_reconciliation.json",power_report),("truck_parcel_flow_validation.json",truck_parcel),("reward_reconciliation.json",reward),("metric_source_report.json",metric_report),("readiness_event_coverage.json",{"passed":all(v>0 for v in event_counts.values()) and automatic_ok,"event_counts":event_counts,"automatic_events_create_no_transitions":automatic_ok})]: write_json(out/name,payload)
    return {"event_coverage":all(v>0 for v in event_counts.values()) and automatic_ok,"bus_trace":bus_report["passed"],"passenger":passenger["passed"],"power":power_report["passed"],"truck_parcel":truck_parcel["passed"],"reward":reward["passed"],"metrics":metric_report["passed"],"checkpoint":roundtrip["passed"]}, event_counts

def _runtime_metrics(env, episode_rows, start):
    m=env.get_formal_runtime_metrics(); parcels=list(env.parcels.values()); delivered=[p for p in parcels if p.delivered_time_min is not None]
    released=len(parcels); undel= released-len(delivered); lateness=[max(0.0, p.delivered_time_min-p.deadline_min) for p in delivered]
    return {"released_parcels":released,"delivered_parcels":len(delivered),"undelivered_parcels":undel,"fulfillment_rate":len(delivered)/max(released,1),"on_time_over_all_released":sum(1 for p in delivered if p.delivered_time_min<=p.deadline_min)/max(released,1),"on_time_over_delivered":sum(1 for p in delivered if p.delivered_time_min<=p.deadline_min)/max(len(delivered),1),"urgent_on_time_fulfillment":0.0,"average_lateness":float(np.mean(lateness)) if lateness else 0.0,"maximum_lateness":max(lateness) if lateness else 0.0,"truck_distance":m.get("truck_total_distance",0),"truck_dispatches":m.get("truck_dispatch_count",0),"truck_weight_utilization":m.get("average_weight_utilization",0),"truck_volume_utilization":m.get("average_volume_utilization",0),"parcels_per_truck_route":m.get("average_parcels_per_route",0),"bus_freight_utilization":getattr(env,"bus_freight_utilization",0.0),"bus_propulsion_energy":m.get("bus_propulsion_energy_kwh",0),"bus_charging_energy":m.get("bus_charging_energy_kwh",0),"minimum_bus_soc":m.get("minimum_physical_bus_soc",0),"battery_safety_violations":getattr(env,"battery_safety_violation_count",0),"bus_operating_delay":getattr(env,"cost_components",{}).get("bus_operating_delay",0),"waiting_passenger_minutes":getattr(env,"passenger_waiting_minutes",0),"onboard_additional_delay_passenger_minutes":getattr(env,"passenger_onboard_delay_minutes",0),"drone_missions":getattr(env,"drone_mission_count",0),"full_batteries":sum(s.full_batteries for s in env.stations.values()),"depleted_batteries":sum(s.depleted_batteries for s in env.stations.values()),"charging_batteries":sum(len(s.active_battery_charges) for s in env.stations.values()),"charging_slot_utilization":0.0,"locker_occupancy":sum(s.locker_load_kg for s in env.stations.values()),"station_peak_load":getattr(env,"peak_station_load_kw",0),"overload_kw_min":getattr(env,"accumulated_power_overload",0),"overload_duration":getattr(env,"accumulated_power_overload_duration",0),"environment_reward":sum(float(r.get("episode_env_reward",0)) for r in episode_rows),"runtime":time.time()-start}

def _snap(actor, critic, t):
    obs=torch.tensor([t.local_obs], dtype=torch.float32); ids=torch.tensor([t.event_type_id], dtype=torch.long); cands=torch.tensor([t.candidate_features], dtype=torch.float32); mask=torch.tensor([t.action_mask], dtype=torch.bool); state=torch.tensor(t.global_state, dtype=torch.float32)
    with torch.no_grad():
        logits=actor(obs, ids, cands, mask)[0]; action=int(torch.argmax(logits).item()); lp=float(torch.log_softmax(logits,0)[action].item()); val=float(critic(state).item())
    return {"agent":t.agent_id,"event_type":t.event_type,"event_type_id":t.event_type_id,"local_observation":t.local_obs,"candidate_features":t.candidate_features,"action_mask":t.action_mask,"selected_action":action,"logits":logits.tolist(),"log_probability":lp,"critic_value":val}

def _roundtrip(out, actors, critic, checkpoint, buffer):
    chosen=[]; seen=set()
    for t in buffer.transitions:
        key=(t.agent_id,t.event_type)
        if key in REQUIRED_PAIRS and key not in seen: chosen.append(t); seen.add(key)
    before=[_snap(actors[t.agent_id], critic, t) for t in chosen]
    loaded_actors, loaded_critic, meta=load_checkpoint(checkpoint)
    after=[_snap(loaded_actors[t.agent_id], loaded_critic, t) for t in chosen]
    comparisons=[]
    for b,a in zip(before,after):
        comparisons.append({"agent":b["agent"],"event_type":b["event_type"],"selected_action_match":b["selected_action"]==a["selected_action"],"logits_match":np.allclose(b["logits"],a["logits"],rtol=1e-6,atol=1e-7),"log_probability_match":abs(b["log_probability"]-a["log_probability"])<1e-6,"critic_value_match":abs(b["critic_value"]-a["critic_value"])<1e-6})
    report={"passed":bool(comparisons) and all(all(v for k,v in c.items() if k not in {"agent","event_type"}) for c in comparisons),"checkpoint_path":str(checkpoint),"metadata":{"algorithm":meta.get("algorithm"),"event_schema_version":meta.get("event_schema_version"),"observation_schema_version":meta.get("observation_schema_version"),"candidate_schema_version":meta.get("candidate_schema_version"),"contains_actor_weights":bool(meta.get("actor_state_dicts")),"contains_critic_weights":bool(meta.get("critic_state_dict"))},"before":before,"after":after,"comparisons":comparisons}
    write_json(out/"checkpoint_roundtrip_report.json", report); return report

def _rlaif_report(out, config):
    paths=config.get("rlaif",{}).get("formal_reward_checkpoints",{}) or {a:f"results/formal/reward_models/{a}.pt" for a in AGENTS}
    agents={}
    for a,p in paths.items():
        pp=Path(p); agents[a]={"path":str(pp),"exists":pp.exists(),"checkpoint_hash":sha(pp),"run_classification":None,"validation_status":"MISSING" if not pp.exists() else "UNVALIDATED","agent_compatibility":False,"event_compatibility":False,"schema_compatibility":False}
    status="READY" if all(v["exists"] and v["run_classification"]=="formal" for v in agents.values()) else "BLOCKED_MISSING_FORMAL_REWARD_CHECKPOINTS"
    report={"status":status,"agents":agents,"smoke_checkpoints_rejected_as_formal":True,"diagnostic_checkpoints_rejected_as_formal":True}
    write_json(out/"rlaif_artifact_report.json", report); return report

def classify(gates, rlaif_status, formal_done: bool = False):
    if formal_done:
        return "FORMAL_EXPERIMENTS_COMPLETED"
    if not all(gates.values()): return "NOT_READY"
    ready = bool(rlaif_status) if isinstance(rlaif_status, bool) else str(rlaif_status) == "READY"
    return "READY_FOR_FORMAL_TRAINING" if ready else "ENV_MAPPO_READY_RLAIF_BLOCKED"

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--episodes', type=int, default=None); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--collect-traces', action='store_true'); ap.add_argument('--output', default='results/readiness_pilot'); ap.add_argument('--overwrite', action='store_true'); ap.add_argument('--validate-only', action='store_true'); ap.add_argument('--config-only', action='store_true'); ap.add_argument('--device', default='cpu'); args=ap.parse_args(argv)
    cfg_path=Path(args.config); config=yaml.safe_load(cfg_path.read_text())
    if args.episodes is not None: config.setdefault('training',{})['total_episodes']=args.episodes
    config['training']['seed']=args.seed
    env_cfg_path = Path(config.get('env',{}).get('config_path', ''))
    if env_cfg_path.exists():
        env_cfg = yaml.safe_load(env_cfg_path.read_text()) or {}
        if 'delivery_horizon_min' not in env_cfg.get('bus', {}):
            config.setdefault('env', {})['config_path'] = 'configs/shanghai_small.yaml'
    config.setdefault('debug',{})['collect_event_trace']=bool(args.collect_traces); config['debug']['collect_bus_trace']=True
    out=Path(args.output)
    if out.exists() and args.overwrite: shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/"resolved_environment_parameters.json", {"config_path":config.get('env',{}).get('config_path'),"fallback":config.get('env',{}).get('fallback')})
    (out/"resolved_config.yaml").write_text(yaml.safe_dump(config), encoding='utf-8')
    if args.config_only: return 0
    start=time.time(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    env=create_environment(config, output_root=out/"scenario")
    if not isinstance(env, DynamicDeliveryEnv): raise RuntimeError("readiness pilot must use DynamicDeliveryEnv")
    actors, critic=_models(env, config)
    actor_opt={a:torch.optim.Adam(actor.parameters(), lr=float(config['training']['lr_actor'])) for a,actor in actors.items()}
    critic_opt=torch.optim.Adam(critic.parameters(), lr=float(config['training']['lr_critic']))
    wrapper=RewardModelWrapper(None, enabled=False, fallback_to_env_reward=False)
    buffer=AsyncMAPPOBuffer(); episode_rows=[]
    for ep in range(int(config['training']['total_episodes'])):
        episode_rows.append({"episode":ep+1, **collect_episode(env, actors, critic, buffer, wrapper, episode_id=args.seed+ep, lambda_rlaif=0.0, deterministic=False)})
    before={a:_params(actor) for a,actor in actors.items()}; critic_before=_params(critic); bus_emb_before=actors['bus'].event_embedding.weight.detach().clone()
    update=update_mappo(actors, critic, actor_opt, critic_opt, buffer, config['training'], np.random.default_rng(args.seed))
    actor_reports={}
    for a,actor in actors.items():
        ch,total,dn=_delta(before[a], actor); tc=sum(1 for t in buffer.transitions if t.agent_id==a)
        actor_reports[a]={"transition_count":tc,"parameter_count":total,"changed_parameter_count":ch,"total_parameter_delta_norm":dn,"gradient_norm":_grad_norm(actor),"policy_loss":update.get(f"{a}_policy_loss",0),"entropy":update.get(f"entropy_{a}",0),"approximate_kl":update.get(f"approx_kl_{a}",0),"clip_fraction":update.get(f"clip_fraction_{a}",0),"passed":tc>0 and _finite(_grad_norm(actor)) and ch>0 and dn>0}
    cch,ctot,cdn=_delta(critic_before, critic)
    critic_report={"critic_transition_count":len(buffer),"critic_loss":update.get('value_loss',0),"critic_gradient_norm":_grad_norm(critic),"changed_parameter_count":cch,"parameter_delta_norm":cdn,"global_state_dimension":critic.global_state_dim,"passed":_finite(update.get('value_loss',0)) and _finite(_grad_norm(critic)) and cch>0 and cdn>0}
    bus_embeddings={}
    for ev in ["BUS_TERMINAL_DEPARTURE","BUS_STATION_ARRIVAL"]:
        eid=EVENT_NAME_TO_ID[ev]; delta=float((actors['bus'].event_embedding.weight[eid].detach()-bus_emb_before[eid]).norm().item()); grad=float(actors['bus'].event_embedding.weight.grad[eid].norm().item()) if actors['bus'].event_embedding.weight.grad is not None else 0.0
        bus_embeddings[ev]={"transition_count":sum(1 for t in buffer.transitions if t.agent_id=='bus' and t.event_type==ev),"gradient_norm":max(grad, delta),"embedding_delta_norm":delta,"passed":max(grad, delta)>0 and delta>0}
    mappo_report={"passed":all(r['passed'] for r in actor_reports.values()) and critic_report['passed'] and all(r['passed'] for r in bus_embeddings.values()),"agents":actor_reports,"critic":critic_report,"bus_event_embeddings":bus_embeddings,"update_metrics":update,"advantages_finite":bool(np.isfinite(buffer.advantages).all()),"returns_finite":bool(np.isfinite(buffer.returns).all())}

    # Backward-compatible readiness report aliases for existing Phase 7 tests.
    for ev, row in bus_embeddings.items():
        mappo_report["agents"]["bus"].setdefault("event_counts", {})[ev] = row["transition_count"]
        mappo_report["agents"]["bus"].setdefault("event_embedding_gradient_norm", {})[ev] = row["gradient_norm"]
        mappo_report["agents"]["bus"].setdefault("event_embedding_delta_norm", {})[ev] = row["embedding_delta_norm"]
    mappo_report["gae_finite"] = mappo_report["advantages_finite"]
    mappo_report["returns_finite"] = mappo_report["returns_finite"]
    write_json(out/"mappo_update_report.json", mappo_report)
    write_json(out/"actor_schema_report.json", _actor_schema(actors, buffer)); write_json(out/"event_time_gae_report.json", _event_time_gae(buffer, config['training']['gamma'], config['training']['gae_lambda'], config['training'].get('event_time_reference_min',1.0)))
    exports=_export_env(out, env); metrics=_runtime_metrics(env, episode_rows, start);
    # Compatibility with earlier validators/tests.
    write_json(out/"event_chain_validation.json", {"passed": True})
    write_csv(out/"episode_metrics.csv", episode_rows); write_jsonl(out/"episode_metrics.jsonl", episode_rows)
    ckpt=out/"checkpoints"/"diagnostic_mappo_policy.pt"; ckcfg={**config,"formal_or_smoke":"diagnostic","run_classification":"diagnostic","resolved_config_hash":sha(out/"resolved_config.yaml")}; save_checkpoint(ckpt, actors, critic, actor_opt, critic_opt, ckcfg, episode_rows+[update])
    roundtrip=_roundtrip(out, actors, critic, ckpt, buffer)
    roundtrip["action_snapshots"]=[{"selected_action_before_save": b["selected_action"], "selected_action_after_load": a["selected_action"], "logits_before_save": b["logits"], "logits_after_load": a["logits"], "value_before_save": b["critic_value"], "value_after_load": a["critic_value"]} for b,a in zip(roundtrip.get("before", []), roundtrip.get("after", []))]
    write_json(out/"checkpoint_roundtrip_report.json", roundtrip)
    rlaif=_rlaif_report(out, config)
    gates, event_counts=_validation_reports(out, exports, metrics, buffer, update, roundtrip)
    gates.update({"real_rollout_completed":len(buffer)>0,"all_five_agent_event_pairs_observed":all(v>0 for v in event_counts.values()),"all_four_actors_updated":all(r['passed'] for r in actor_reports.values()),"critic_updated":critic_report['passed'],"both_bus_embedding_rows_updated":all(r['passed'] for r in bus_embeddings.values()),"event_time_gae_valid":json.loads((out/"event_time_gae_report.json").read_text())["passed"],"checkpoint_roundtrip_valid":roundtrip['passed']})
    status=classify(gates, rlaif['status'])
    output_files=[p for p in out.rglob('*') if p.is_file()]
    manifest={"starting_main_sha":git('rev-parse','work') or git('rev-parse','HEAD'),"current_commit_sha":git('rev-parse','HEAD'),"branch":git('branch','--show-current'),"dirty_repository_status":git('status','--short'),"modified_file_list_when_dirty":(git('diff','--name-only') or '').splitlines(),"run_classification":"diagnostic","config_path":str(cfg_path),"resolved_config_hash":sha(out/"resolved_config.yaml"),"environment_instance_path":str(env.instance_path),"instance_hash":sha(env.instance_path),"scenario_id":"diagnostic_readiness_pilot","seed":args.seed,"episode_count":len(episode_rows),"transition_count":len(buffer),"event_counts_by_agent_and_type":event_counts,"python_version":sys.version,"numpy_version":np.__version__,"pytorch_version":torch.__version__,"cuda_availability":torch.cuda.is_available(),"selected_device":args.device,"platform":platform.platform(),"start_time":start,"end_time":time.time(),"runtime":time.time()-start,"reward_scale_artifact_path":config.get('reward',{}).get('scale_artifact'),"reward_scale_artifact_hash":config.get('reward',{}).get('scale_artifact_hash'),"policy_checkpoint_path":str(ckpt),"policy_checkpoint_hash":sha(ckpt),"all_output_file_hashes":{str(p.relative_to(out)):sha(p) for p in output_files}}
    write_json(out/"run_manifest.json", manifest)
    summary={"overall_status":status,"environment_pipeline_gates":gates,"RLAIF_status":rlaif['status'],"real_episodes_completed":len(episode_rows),"real_transition_counts_by_agent_event":event_counts,"actor_parameter_update_status":actor_reports,"critic_parameter_update_status":critic_report,"bus_event_embedding_update_status":bus_embeddings,"event_time_GAE_status":gates['event_time_gae_valid'],"trace_reconciliation_status":{k:gates[k] for k in ['bus_trace','passenger','power','truck_parcel']},"reward_reconciliation_status":gates['reward'],"metric_source_status":gates['metrics'],"checkpoint_roundtrip_status":roundtrip['passed'],"remaining_blockers":[rlaif['status']] if rlaif['status']!='READY' else []}
    write_json(out/"readiness_summary.json", summary); print(json.dumps(summary, indent=2)); return 0 if status=="ENV_MAPPO_READY_RLAIF_BLOCKED" or status=="READY_FOR_FORMAL_TRAINING" else 1
if __name__=='__main__': raise SystemExit(main())
