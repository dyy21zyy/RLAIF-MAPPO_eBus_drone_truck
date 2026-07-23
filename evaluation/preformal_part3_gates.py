"""Pre-formal Part 3 benchmark, ablation, sensitivity, pairing, readiness gates."""
from __future__ import annotations

import hashlib, json, math, os, subprocess, sys, time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Callable

from evaluation.formal_policy_registry import get_formal_policy_spec, LEARNED_METHODS
from evaluation.experiment_aggregation import aggregate_compatible, paired_differences as production_paired_differences

CANONICAL_EVENTS = ("PARCEL_RELEASE","TRUCK_AVAILABLE","BUS_TERMINAL_DEPARTURE","BUS_STATION_ARRIVAL","STATION_OPERATION")
REQUIRED_METRICS = (
 "released_parcels","delivered_parcels","undelivered_parcels","fulfillment_rate","on_time_delivered_parcels","on_time_rate_over_all_released","on_time_rate_over_delivered","urgent_parcels_released","urgent_parcels_delivered_on_time","urgent_on_time_fulfillment","average_lateness","maximum_lateness",
 "truck_dispatch_count","truck_route_count","truck_distance","truck_travel_time","truck_weight_utilization","truck_volume_utilization","parcels_per_route","truck_cost",
 "bus_freight_load","bus_freight_utilization","bus_charging_count","bus_charging_energy","bus_propulsion_energy","minimum_bus_soc","bus_battery_violation_count","bus_operating_delay","ordinary_stops_visited","integrated_stations_visited",
 "waiting_passenger_minutes","onboard_additional_delay_passenger_minutes","passengers_boarded","passengers_alighted","remaining_passenger_queues",
 "drone_missions","full_batteries","depleted_batteries","charging_batteries","charging_slot_utilization","locker_occupancy","station_peak_load","overload_kw_min","overload_duration","battery_charging_energy",
 "environment_reward","assignment_rlaif_contribution","truck_rlaif_contribution","bus_rlaif_contribution","station_rlaif_contribution","total_weighted_rlaif_reward","combined_reward","infeasible_action_count","reward_fallback_count","decision_count_by_agent","transition_count","runtime",
)
ARTIFACT_KEYS=("road_network_hash","bus_route_hash","timetable_hash","parcel_hash","passenger_demand_hash","station_load_hash","distance_matrix_hash","distance_matrix_hashes")

class BenchmarkIntegrityError(RuntimeError): pass
class PreformalPairedScenarioMismatchError(RuntimeError): pass
class PreformalConfigDifferenceError(RuntimeError): pass

def sha256_file(path: str|Path) -> str|None:
    p=Path(path); return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None

def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()

def run_command_recorded(command: list[str], *, cwd: str|Path, output_dir: str|Path, config_path: str|Path|None=None) -> dict[str,Any]:
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    start=time.time(); stdout=out/'stdout.txt'; stderr=out/'stderr.txt'
    proc=subprocess.run(command,cwd=str(cwd),text=True,capture_output=True,check=False)
    end=time.time(); stdout.write_text(proc.stdout); stderr.write_text(proc.stderr)
    rec={"command":command,"command_text":" ".join(command),"working_directory":str(cwd),"start_time":start,"end_time":end,"runtime":end-start,"return_code":proc.returncode,"stdout_path":str(stdout),"stderr_path":str(stderr),"resolved_config_hash":sha256_file(config_path) if config_path else None}
    (out/'command_record.json').write_text(json.dumps(rec,indent=2,sort_keys=True))
    return rec

def invoke_production_benchmark(config_path: str|Path, output_root: str|Path, *, cwd: str|Path|None=None) -> dict[str,Any]:
    return run_command_recorded([sys.executable,"-m","experiments.run_paper_benchmark","--config",str(config_path),"--output-root",str(output_root),"--continue-on-error"], cwd=cwd or Path.cwd(), output_dir=Path(output_root)/"preformal_command", config_path=config_path)

def invoke_production_ablation(config_path: str|Path, output_root: str|Path, *, cwd: str|Path|None=None) -> dict[str,Any]:
    return run_command_recorded([sys.executable,"-m","experiments.run_ablation_matrix","--config",str(config_path),"--output-root",str(output_root)], cwd=cwd or Path.cwd(), output_dir=Path(output_root)/"preformal_command", config_path=config_path)

def invoke_production_sensitivity(config_path: str|Path, output_root: str|Path, *, cwd: str|Path|None=None) -> dict[str,Any]:
    return run_command_recorded([sys.executable,"-m","experiments.run_sensitivity_matrix","--config",str(config_path),"--output-root",str(output_root)], cwd=cwd or Path.cwd(), output_dir=Path(output_root)/"preformal_command", config_path=config_path)

def method_matrix(artifacts: dict[str,Any]) -> list[dict[str,Any]]:
    rows=[]
    for mid in ("truck_direct_heuristic","integrated_rule_based","assignment_ppo","mappo_env","mappo_rlaif_assignment","mappo_rlaif_all"):
        spec=get_formal_policy_spec(mid); ck=artifacts.get(mid) or artifacts.get(f"{mid}_checkpoint")
        if mid in LEARNED_METHODS and not ck:
            rows.append({"method_id":mid,"enabled":False,"status":"blocked_missing_artifact","policy_checkpoint":None,"rlaif_scope":spec.expected_rlaif_scope,"enabled_reward_agents":list(spec.enabled_reward_agents)})
        else:
            rows.append({"method_id":mid,"enabled":True,"status":"ready","policy_checkpoint":ck,"reward_checkpoint":None if not spec.enabled_reward_agents else artifacts.get(f"{mid}_reward_checkpoints",{}),"rlaif_scope":spec.expected_rlaif_scope,"enabled_reward_agents":list(spec.enabled_reward_agents)})
    return rows

def _metric_entry(metrics:dict,k:str)->dict:
    v=metrics.get(k)
    if isinstance(v,dict): return v
    return {"value":v,"available":v is not None,"finite":isinstance(v,(int,float)) and math.isfinite(v),"source":"runtime" if v is not None else None,"formula":None,"legitimate_zero":v==0}

def validate_benchmark_row(row:dict, *, strict=True) -> None:
    if row.get('status')=='success' and int(row.get('transition_count') or 0) <= 0: raise BenchmarkIntegrityError('successful row requires transition_count > 0 and env.step evidence')
    if row.get('status')!='success': return
    if not row.get('env_constructed', True) or not row.get('env_reset_called', True) or not row.get('env_step_called', row.get('transition_count',0)>0): raise BenchmarkIntegrityError('successful row lacks real environment runtime evidence')
    metrics=row.get('formal_metrics') or row.get('metrics') or {}
    for k in REQUIRED_METRICS:
        e=_metric_entry(metrics,k)
        if strict and (not e.get('available') or not e.get('finite') or not e.get('source')): raise BenchmarkIntegrityError(f'missing required metric: {k}')
    if all((_metric_entry(metrics,k).get('value') in (0,0.0,None)) for k in REQUIRED_METRICS): raise BenchmarkIntegrityError('fixed zero placeholder row rejected')

def validate_reconciliation(row:dict)->None:
    m=row.get('formal_metrics') or row.get('metrics') or {}; val=lambda k: _metric_entry(m,k).get('value') or 0
    if val('delivered_parcels')+val('undelivered_parcels') != val('released_parcels'): raise BenchmarkIntegrityError('parcel accounting mismatch')
    if val('on_time_delivered_parcels') > val('delivered_parcels') or val('urgent_parcels_delivered_on_time') > val('urgent_parcels_released'): raise BenchmarkIntegrityError('on-time parcel bounds mismatch')
    for k in ('truck_weight_utilization','truck_volume_utilization','bus_freight_utilization','charging_slot_utilization','locker_occupancy'):
        x=val(k)
        if x < 0 or x > 1: raise BenchmarkIntegrityError(f'utilization out of bounds: {k}')
    if abs(val('total_weighted_rlaif_reward')-(val('assignment_rlaif_contribution')+val('truck_rlaif_contribution')+val('bus_rlaif_contribution')+val('station_rlaif_contribution'))) > 1e-9: raise BenchmarkIntegrityError('RLAIF reward reconciliation mismatch')
    if abs(val('combined_reward')-(val('environment_reward')+val('total_weighted_rlaif_reward'))) > 1e-9: raise BenchmarkIntegrityError('combined reward reconciliation mismatch')
    if val('reward_fallback_count') != 0: raise BenchmarkIntegrityError('reward fallback count must be zero')
    if row.get('method_id')=='mappo_env' and any(val(k)!=0 for k in ('assignment_rlaif_contribution','truck_rlaif_contribution','bus_rlaif_contribution','station_rlaif_contribution')): raise BenchmarkIntegrityError('environment MAPPO must have zero RLAIF')
    if row.get('method_id')=='mappo_rlaif_assignment' and any(val(k)!=0 for k in ('truck_rlaif_contribution','bus_rlaif_contribution','station_rlaif_contribution')): raise BenchmarkIntegrityError('assignment RLAIF leaked to other agents')

def expected_row_report(methods:list[dict], scenarios:list[str]) -> dict:
    exp=[]
    for m in methods:
        if not m.get('enabled',True): continue
        seeds=m.get('training_seeds',[m.get('training_seed')])
        if m.get('method_id') not in LEARNED_METHODS: seeds=[None]
        for s in seeds:
            for sc in scenarios: exp.append((m.get('method_id'),s,sc))
    return {"expected_rows":len(exp),"expected_identities":exp}

def validate_expected_rows(methods, scenarios, rows):
    rep=expected_row_report(methods,scenarios); actual=[(r.get('method_id'),r.get('training_seed'),r.get('scenario_id')) for r in rows if r.get('status')=='success']; failed=[r for r in rows if r.get('status')!='success']
    rep.update({"actual_successful_rows":len(actual),"actual_failed_rows":len(failed),"missing_rows":[x for x in rep['expected_identities'] if x not in actual],"duplicate_rows":[x for x in set(actual) if actual.count(x)>1]}); return rep

def validate_event_coverage(rows):
    counts={e:0 for e in CANONICAL_EVENTS}; by_method={}; by_scenario={}; decisions={}
    for r in rows:
        for e,c in (r.get('event_counts') or {}).items():
            if e in counts: counts[e]+=c; by_method.setdefault(r.get('method_id'),{}).setdefault(e,0); by_method[r.get('method_id')][e]+=c; by_scenario.setdefault(r.get('scenario_id'),{}).setdefault(e,0); by_scenario[r.get('scenario_id')][e]+=c
        for a,c in (r.get('decision_count_by_agent') or _metric_entry((r.get('formal_metrics') or {}),'decision_count_by_agent').get('value') or {}).items(): decisions[a]=decisions.get(a,0)+c
    missing=[e for e,c in counts.items() if c<=0]
    if missing: raise BenchmarkIntegrityError('missing canonical event coverage: '+', '.join(missing))
    return {"event_counts":counts,"event_counts_by_method":by_method,"event_counts_by_scenario":by_scenario,"decision_counts_by_agent":decisions}

def scenario_signature(row:dict, *, sensitivity=False):
    art=row.get('artifact_hashes') or row.get('exogenous_artifact_hashes') or {}
    sig=[row.get('scenario_id'),row.get('scenario_content_hash'),row.get('instance_hash'),row.get('scenario_manifest_hash'),row.get('scenario_bank_hash'),tuple(sorted((k,art.get(k)) for k in sorted(set(ARTIFACT_KEYS)|set(art))))]
    if sensitivity: sig.append(row.get('scenario_family_id'))
    return tuple(sig)

def assert_preformal_pairable(a,b,*,sensitivity=False):
    if scenario_signature(a,sensitivity=sensitivity)!=scenario_signature(b,sensitivity=sensitivity): raise PreformalPairedScenarioMismatchError('paired scenario identity mismatch')
    return True

def paired_statistics(rows, baseline_selector, comparison_selector, metrics=('environment_reward',), *, sensitivity=False):
    out={"publication_eligible":False,"metrics":{}}
    base={scenario_signature(r,sensitivity=sensitivity):r for r in rows if baseline_selector(r) and r.get('status')=='success'}
    for metric in metrics:
        pairs=[]
        for r in rows:
            if comparison_selector(r) and r.get('status')=='success':
                sig=scenario_signature(r,sensitivity=sensitivity)
                if sig in base:
                    bv=_metric_entry(base[sig].get('formal_metrics') or base[sig],metric).get('value'); cv=_metric_entry(r.get('formal_metrics') or r,metric).get('value')
                    pairs.append({'baseline_value':bv,'comparison_value':cv,'paired_difference':cv-bv})
        vals=[p['paired_difference'] for p in pairs]; summary={'paired_sample_count':len(vals),'status':'insufficient_samples'}
        if len(vals)>1:
            sd=stdev(vals); se=sd/math.sqrt(len(vals)); summary.update({'mean_paired_difference':fmean(vals),'standard_deviation':sd,'standard_error':se,'ci95_low':fmean(vals)-1.96*se,'ci95_high':fmean(vals)+1.96*se,'status':'ok'})
        out['metrics'][metric]={'pairs':pairs,'summary':summary}
    return out

def validate_checkpoint_separation(variants:list[dict], *, strict=True):
    seen={}
    for v in variants:
        if not v.get('retraining_required'): continue
        for k in ('resolved_training_config_hash','checkpoint_path','checkpoint_hash','training_log','evaluation_dir'):
            if not v.get(k): raise BenchmarkIntegrityError(f'ablation missing {k}')
        h=v['checkpoint_hash']
        if h in seen and (strict or not v.get('diagnostic_dummy_artifact')): raise BenchmarkIntegrityError('duplicate checkpoint hash across retraining-required variants')
        seen[h]=v.get('variant_id')
    return True

def validate_sensitivity_protocol(rows:list[dict], protocol:str):
    if protocol=='fixed_policy_robustness' and len({r.get('policy_checkpoint_hash') for r in rows})!=1: raise BenchmarkIntegrityError('fixed-policy sensitivity requires identical checkpoint hash')
    if protocol=='retrained_policy_sensitivity' and len({r.get('policy_checkpoint_hash') for r in rows})!=len({r.get('factor_value') for r in rows}): raise BenchmarkIntegrityError('retrained sensitivity requires separate checkpoints per value')
    if len({r.get('protocol') for r in rows})>1: raise BenchmarkIntegrityError('fixed and retrained sensitivity protocols must remain separate')
    return True

def validate_config_differences(base:dict, comp:dict, allowed_paths:set[str]) -> dict:
    diffs=[]
    def walk(a,b,p=''):
        for k in set((a or {}).keys())|set((b or {}).keys()):
            q=f'{p}.{k}' if p else k
            if isinstance((a or {}).get(k),dict) and isinstance((b or {}).get(k),dict): walk(a[k],b[k],q)
            elif (a or {}).get(k)!=(b or {}).get(k): diffs.append(q)
    walk(base,comp); unexpected=[d for d in diffs if d not in allowed_paths and not any(d.startswith(x+'.') for x in allowed_paths)]
    if unexpected: raise PreformalConfigDifferenceError('unexpected config differences: '+', '.join(unexpected))
    return {'intended_config_differences':[d for d in diffs if d not in unexpected],'unexpected_config_differences':unexpected}

def readiness_status(stage_statuses:dict, *, run_classification='diagnostic', full_test_suite_status='passed'):
    if stage_statuses.get('benchmark_execution')!='passed': return 'BLOCKED_BENCHMARK'
    if run_classification=='diagnostic' and all(v=='passed' for v in stage_statuses.values()): return 'PREFORMAL_DIAGNOSTIC_PASSED'
    if stage_statuses.get('environment_mappo_training')=='passed' and stage_statuses.get('assignment_rlaif_mappo_training')!='passed': return 'PREFORMAL_ENVIRONMENT_PATH_PASSED_RLAIF_BLOCKED'
    if stage_statuses.get('assignment_rlaif_mappo_training')=='passed' and stage_statuses.get('full_rlaif_mappo_training') in {'blocked_missing_artifact','skipped_optional','failed'}: return 'PREFORMAL_ASSIGNMENT_RLAIF_PASSED_FULL_RLAIF_BLOCKED'
    if all(v in {'passed','skipped_optional'} for v in stage_statuses.values()) and full_test_suite_status=='passed': return 'PREFORMAL_ALL_REQUIRED_PATHS_PASSED'
    return 'PREFORMAL_DIAGNOSTIC_FAILED' if run_classification=='diagnostic' else 'BLOCKED_FULL_TEST_SUITE'
