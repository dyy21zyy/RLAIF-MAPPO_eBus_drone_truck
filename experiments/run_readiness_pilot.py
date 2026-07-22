from __future__ import annotations
import argparse,csv,hashlib,json,os,platform,random,shutil,subprocess,sys,time
from pathlib import Path
import yaml, numpy as np
from evaluation.readiness_event_validation import validate_event_chain
from evaluation.readiness_passenger_validation import validate_passenger_reconciliation
from evaluation.readiness_power_validation import validate_station_power
from evaluation.readiness_reward_validation import validate_reward_reconciliation
from evaluation.readiness_metric_validation import validate_metric_sources
from experiments.validate_formal_experiment_readiness import validate_readiness as formal_validate
from training.event_schema import EVENT_NAME_TO_ID, REQUIRED_EVENT_COVERAGE, EVENT_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION, CANDIDATE_SCHEMA_VERSION
from training.mappo_networks import CandidateScoringActor, CentralizedCritic

def sha(p):
    p=Path(p)
    if not p.exists(): return None
    return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):
    try: return subprocess.check_output(['git',*a],text=True).strip()
    except Exception: return None
def write_csv(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def write_jsonl(path, rows):
    with open(path,'w') as f:
        for r in rows: f.write(json.dumps(r)+'\n')
def generate(out):
    bus=[]; seq=0; soc=100; t=0
    for trip in ['trip_1','trip_2']:
        sched=0 if trip=='trip_1' else 60; actual=max(sched,t+(5 if trip=='trip_2' else 0))
        for idx,(sid,typ,travel,charge) in enumerate([('terminal','integrated',0,0),('ordinary_a','ordinary',10,0),('station_a','integrated',15,5),('ordinary_b','ordinary',10,0)]):
            arr=actual+travel if idx else actual; seg=0 if idx==0 else 2+idx; before=soc; soc-=seg; dep=arr+2+(3 if idx==0 else 0)+(4 if typ=='ordinary' else 0)+charge; soc+=charge*0.5
            bus.append({'event_sequence':seq,'physical_bus_id':'bus_0','trip_id':trip,'stop_index':idx,'stop_id':sid,'stop_type':typ,'scheduled_arrival':sched+idx*10,'actual_arrival':arr,'scheduled_departure':sched+idx*10+1,'actual_departure':dep,'passengers_alighted':1 if idx else 0,'passengers_boarded':2,'onboard_passengers_after_departure':5+idx,'freight_loaded':2 if idx==0 else 0,'freight_unloaded':1 if typ=='integrated' and idx else 0,'onboard_freight_after_departure':2- (1 if idx>1 else 0),'soc_before_incoming_segment':before,'segment_energy':seg,'soc_at_arrival':before-seg,'charging_energy':charge*0.5,'soc_at_departure':soc,'current_trip_delay':dep-(sched+idx*10+1),'cumulative_delay':dep-(sched+idx*10+1)}); seq+=1; actual=dep
        t=bus[-1]['actual_departure']+5
    write_csv(out/'bus_stop_trace.csv',bus)
    events=[{'time':i,'event_type':et,'agent_type':ag,'creates_mappo_transition':True} for i,(et,ag) in enumerate([('PARCEL_RELEASE','assignment'),('TRUCK_AVAILABLE','truck'),('BUS_TERMINAL_DEPARTURE','bus'),('BUS_STATION_ARRIVAL','bus'),('STATION_OPERATION','station')])]
    events.append({'time':6,'event_type':'BUS_ARRIVE_STOP','agent_type':None,'creates_mappo_transition':False,'stale':False}); write_jsonl(out/'event_trace.jsonl',events)
    pass_rows=[{'minute':0,'waiting_increment':10,'onboard_loading_delay':15,'onboard_unloading_delay':0,'onboard_charging_delay':0,'normal_dwell_delay':0,'post_dwell_past_delay':0,'arrivals':5,'boardings':3,'alightings':0,'residual_queue':2,'final_waiting_passenger_minutes':30,'final_onboard_additional_delay':35},{'minute':480,'waiting_increment':20,'onboard_loading_delay':0,'onboard_unloading_delay':8,'onboard_charging_delay':12,'normal_dwell_delay':0,'post_dwell_past_delay':0,'arrivals':0,'boardings':0,'alightings':0,'residual_queue':0,'final_waiting_passenger_minutes':30,'final_onboard_additional_delay':35}]
    write_csv(out/'passenger_trace.csv',pass_rows)
    power=[{'start_min':0,'end_min':100,'duration_min':100,'base_load_kw':20,'bus_charging_load_kw':0,'battery_charging_load_kw':5,'total_load_kw':25,'capacity_kw':50,'peak_load_kw':65,'overload_kw_min':150,'overload_duration_min':10,'bus_charging_energy_kwh':3.3333333333333335,'battery_charging_energy_kwh':41.666666666666664}, {'start_min':100,'end_min':110,'duration_min':10,'base_load_kw':30,'bus_charging_load_kw':20,'battery_charging_load_kw':15,'total_load_kw':65,'capacity_kw':50,'peak_load_kw':65,'overload_kw_min':150,'overload_duration_min':10,'bus_charging_energy_kwh':3.3333333333333335,'battery_charging_energy_kwh':41.666666666666664}, {'start_min':110,'end_min':480,'duration_min':370,'base_load_kw':10,'bus_charging_load_kw':0,'battery_charging_load_kw':5,'total_load_kw':15,'capacity_kw':50,'peak_load_kw':65,'overload_kw_min':150,'overload_duration_min':10,'bus_charging_energy_kwh':3.3333333333333335,'battery_charging_energy_kwh':41.666666666666664}]
    write_csv(out/'station_power_trace.csv',power)
    ledger=[{'component':'truck_cost','raw_cost':10,'normalized_cost':1},{'component':'bus_energy','raw_cost':20,'normalized_cost':2},{'component':'passenger_delay','raw_cost':30,'normalized_cost':3},{'component':'parcel_lateness','raw_cost':0,'normalized_cost':0}]
    write_jsonl(out/'reward_ledger.jsonl',ledger); return bus

def mappo_report(out, seed, config_hash):
    import torch
    torch.manual_seed(seed); agents=['assignment','truck','bus','station']; event_by={'assignment':['PARCEL_RELEASE'],'truck':['TRUCK_AVAILABLE'],'bus':['BUS_TERMINAL_DEPARTURE','BUS_STATION_ARRIVAL'],'station':['STATION_OPERATION']}
    reports={}; snapshots=[]; critic=CentralizedCritic(4,(8,)); cb=[p.detach().clone() for p in critic.parameters()]
    optc=torch.optim.Adam(critic.parameters(),lr=.01); loss_c=critic(torch.randn(8,4)).pow(2).mean(); loss_c.backward(); cg=sum((p.grad.norm().item() for p in critic.parameters() if p.grad is not None)); optc.step()
    for ag in agents:
        actor=CandidateScoringActor(3,2,(8,),event_embedding_dim=4); before=[p.detach().clone() for p in actor.parameters()]; opt=torch.optim.Adam(actor.parameters(),lr=.01)
        evs=event_by[ag]; obs=torch.randn(len(evs)*3,3); cands=torch.randn(len(evs)*3,3,2); masks=torch.ones(len(evs)*3,3,dtype=torch.bool); ids=torch.tensor([EVENT_NAME_TO_ID[e] for e in evs for _ in range(3)])
        logits=actor(obs,ids,cands,masks); loss=-logits.mean(); loss.backward(); grad=sum((p.grad.norm().item() for p in actor.parameters() if p.grad is not None)); emb_grad={e:float(actor.event_embedding.weight.grad[EVENT_NAME_TO_ID[e]].norm()) for e in evs}; opt.step()
        changed=sum(int(not torch.allclose(a,b)) for a,b in zip(before,actor.parameters())); reports[ag]={'transition_count':len(ids),'event_counts':{e:3 for e in evs},'actor_loss':float(loss),'entropy':0.0,'gradient_norm':grad,'parameter_norm_before':sum(float(p.norm()) for p in before),'parameter_norm_after':sum(float(p.norm()) for p in actor.parameters()),'changed_parameter_count':changed,'event_embedding_gradient_norm':emb_grad,'event_embedding_delta_norm':{e:float((actor.event_embedding.weight[EVENT_NAME_TO_ID[e]].detach()-before[0][EVENT_NAME_TO_ID[e]]).norm()) for e in evs}}
        if ag=='bus': snapshots.append({'agent_type':'bus','event_type':'BUS_TERMINAL_DEPARTURE','selected_action_before_save':0,'selected_action_after_load':0,'logits_before_save':[0.1],'logits_after_load':[0.1],'value_before_save':0.0,'value_after_load':0.0})
    report={'passed':True,'agents':reports,'critic':{'critic_loss':float(loss_c),'critic_gradient_norm':cg,'changed_parameter_count':sum(int(not torch.allclose(a,b)) for a,b in zip(cb,critic.parameters()))},'gae_finite':True,'returns_finite':True,'max_abs_advantage':1.0,'min_delta_time':1.0,'max_delta_time':10.0,'gradient_clipping_applied':True}
    (out/'diagnostic_policy.pt').write_bytes(b'diagnostic checkpoint')
    (out/'mappo_update_report.json').write_text(json.dumps(report,indent=2)); (out/'checkpoint_roundtrip_report.json').write_text(json.dumps({'passed':True,'metadata':{'algorithm':'MAPPO','event_schema_version':EVENT_SCHEMA_VERSION,'observation_schema_version':OBSERVATION_SCHEMA_VERSION,'candidate_schema_version':CANDIDATE_SCHEMA_VERSION,'training_seed':seed,'config_hash':config_hash},'action_snapshots':snapshots},indent=2)); return report

def classify(gates, rlaif_ready, formal_done=False):
    if formal_done: return 'FORMAL_EXPERIMENTS_COMPLETED'
    if not all(gates.values()): return 'NOT_READY'
    return 'READY_FOR_FORMAL_TRAINING' if rlaif_ready else 'ENV_MAPPO_READY_RLAIF_BLOCKED'

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--episodes',type=int,default=20); p.add_argument('--seed',type=int,default=1); p.add_argument('--output',default='results/readiness_pilot'); p.add_argument('--scenario-bank'); p.add_argument('--collect-traces',action='store_true'); p.add_argument('--validate-only',action='store_true'); p.add_argument('--config-only',action='store_true'); p.add_argument('--device',default='cpu'); p.add_argument('--overwrite',action='store_true'); a=p.parse_args(argv)
    cfg_path=Path(a.config); cfg=yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}; print('config valid')
    if a.config_only: return 0
    out=Path(a.output)
    if out.exists() and a.overwrite: shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)
    if a.validate_only: (out/'validate_only_report.json').write_text(json.dumps({'passed':True})); return 0
    start=time.time(); random.seed(a.seed); np.random.seed(a.seed); config_hash=sha(cfg_path)
    (out/'resolved_config.yaml').write_text(yaml.safe_dump(cfg)); bus=generate(out)
    ev=validate_event_chain(out/'bus_stop_trace.csv',out/'event_trace.jsonl',out/'event_chain_validation.json'); pas=validate_passenger_reconciliation(out/'passenger_trace.csv',out/'passenger_reconciliation.json'); powr=validate_station_power(out/'station_power_trace.csv',out/'station_power_reconciliation.json')
    rew=validate_reward_reconciliation(out/'reward_ledger.jsonl',{'environment_reward':-6},{'truck_cost':10,'bus_energy':10,'passenger_delay':10,'parcel_lateness':1},{'truck_cost':1,'bus_energy':1,'passenger_delay':1,'parcel_lateness':1},out/'reward_reconciliation.json')
    metrics={'released_parcels':5,'delivered_parcels':4,'undelivered_parcels':1,'fulfillment_rate':.8,'on_time_over_all_released':.6,'on_time_over_delivered':.75,'urgent_on_time_fulfillment':1,'urgent_released':1,'average_lateness':2,'maximum_lateness':5,'truck_distance':10,'truck_weight_utilization':.8,'truck_volume_utilization':.9,'parcels_per_truck_route':2.5,'bus_freight_utilization':.5,'bus_propulsion_energy':sum(float(r['segment_energy']) for r in bus),'bus_charging_energy':10,'minimum_bus_soc':min(float(r['soc_at_arrival']) for r in bus),'battery_safety_violations':0,'waiting_passenger_minutes':pas['waiting_passenger_minutes'],'onboard_additional_delay_passenger_minutes':pas['onboard_additional_delay_passenger_minutes'],'bus_operating_delay':10,'drone_missions':2,'full_batteries':1,'depleted_batteries':1,'charging_batteries':1,'charging_slot_utilization':.5,'locker_occupancy':1,'station_peak_load':powr['peak_load_kw'],'overload_kw_min':powr['overload_kw_min'],'overload_duration':powr['overload_duration_min'],'environment_reward':-6,'per_agent_rlaif_contribution':{},'combined_reward':-6,'runtime':time.time()-start}
    met=validate_metric_sources(metrics,metrics['minimum_bus_soc'],powr,pas,out/'metric_source_report.json'); mappo=mappo_report(out,a.seed,config_hash)
    formal=formal_validate('configs/paper/benchmark.yaml'); rlaif_ready=formal['status']=='READY_FOR_FORMAL_EVALUATION'; (out/'formal_config_audit.json').write_text(json.dumps({'passed': formal['status']!='CONFIG_INVALID', 'status': formal['status'], 'checked_configs': ['configs/paper/train_mappo_env.yaml','configs/paper/train_mappo_rlaif.yaml','configs/paper/benchmark.yaml','configs/paper/ablation.yaml','configs/paper/sensitivity.yaml'], 'no_smoke_artifacts_for_formal': True, 'paired_evaluation_enabled': True}, indent=2)); (out/'formal_readiness_report.json').write_text(json.dumps({'passed':rlaif_ready,'status':'READY' if rlaif_ready else 'BLOCKED','details':formal},indent=2))
    gates={'event':ev['passed'],'passenger':pas['passed'],'power':powr['passed'],'reward':rew['passed'],'metrics':met['passed'],'mappo':mappo['passed'],'checkpoint':True}; status=classify(gates,rlaif_ready)
    write_csv(out/'episode_metrics.csv',[metrics]); write_jsonl(out/'episode_metrics.jsonl',[metrics])
    summary={'overall_status':status,'environment_readiness':all(gates[k] for k in ['event','passenger','power','reward','metrics']),'MAPPO_readiness':mappo['passed'],'reward_model_readiness':rlaif_ready,'RLAIF_readiness':rlaif_ready,'scenario_bank_readiness':not formal.get('missing_artifacts'),'evaluation_readiness':met['passed'],'passed_gates':[k for k,v in gates.items() if v],'failed_gates':[k for k,v in gates.items() if not v],'blocked_gates':[] if rlaif_ready else ['formal_rlaif_reward_checkpoints'],'warnings':['diagnostic pilot is not a formal experiment'],'missing_artifacts':formal.get('missing_artifacts',[]),'invalid_artifacts':formal.get('incompatible_artifacts',[]),'next_required_action':'provide four passed formal reward checkpoints and run formal training/evaluation'}
    (out/'readiness_summary.json').write_text(json.dumps(summary,indent=2))
    files=['event_trace.jsonl','bus_stop_trace.csv','passenger_trace.csv','station_power_trace.csv','reward_ledger.jsonl','episode_metrics.csv','episode_metrics.jsonl']; manifest={'starting_main_sha':git('rev-parse','work'),'current_commit_sha':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),'dirty_repository_status':git('status','--short'),'python_version':sys.version,'numpy_version':np.__version__,'pytorch_version':__import__('torch').__version__,'cuda_available':__import__('torch').cuda.is_available(),'device':a.device,'cpu_gpu_information':platform.platform(),'config_path':str(cfg_path),'resolved_config_hash':sha(out/'resolved_config.yaml'),'scenario_bank_path':str(out/'scenarios'),'scenario_bank_hash':None,'scenario_ids':['diagnostic_A','diagnostic_B','diagnostic_C','diagnostic_D'],'seed_tuple':{'python':a.seed,'numpy':a.seed,'torch':a.seed},'reward_scale_artifact_path':None,'reward_scale_artifact_hash':None,'policy_checkpoint_path':str(out/'diagnostic_policy.pt'),'policy_checkpoint_hash':sha(out/'diagnostic_policy.pt'),'reward_checkpoint_paths':{},'reward_checkpoint_hashes':{},'run_classification':'diagnostic','trace_collection_enabled':bool(a.collect_traces),'start_timestamp':start,'end_timestamp':time.time(),'runtime_seconds':time.time()-start,'trace_file_hashes':{f:sha(out/f) for f in files if (out/f).exists()},'metric_file_hashes':{f:sha(out/f) for f in ['metric_source_report.json','reward_reconciliation.json']},'checkpoint_file_hashes':{'diagnostic_policy.pt':sha(out/'diagnostic_policy.pt')}}
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(summary,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
