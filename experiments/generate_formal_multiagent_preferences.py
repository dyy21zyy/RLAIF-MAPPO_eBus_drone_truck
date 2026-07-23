"""Formal four-agent RLAIF preference generation.

Collects normalized decision records, builds feasible same-state candidate pairs,
queries the configured OpenAI-compatible evaluator, validates JSON responses, and
writes one canonical preference JSONL per RLAIF agent. Runtime never fabricates
labels; tests may inject a deterministic evaluator callable.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from rlaif.ai_evaluator import APISettings, _default_api_call
from rlaif.grouped_split import grouped_split
from rlaif.preference_dataset import write_jsonl, read_jsonl
from envs import DynamicDeliveryEnv
from evaluation.scenario_bank import load_scenario_bank, load_frozen_instance, load_bank_manifest, sha256_file as bank_sha256_file
from training.event_schema import (AGENT_TYPES, CANDIDATE_SCHEMA_VERSION, DECISION_EVENT_SPECS,
    EVENT_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION, REQUIRED_EVENT_COVERAGE,
    decision_event_agent, normalize_decision_event_type)

PROMPT_VERSION = "four_agent_consequence_v1"
RESPONSE_SCHEMA_VERSION = "rlaif_preference_json_v1"
FORBIDDEN_RESPONSE_WORDS = ("mappo", "ppo", "rlaif", "policy", "algorithm")


def sha_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

def sha_file(path: Path) -> str:
    h=hashlib.sha256();
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}

def evaluator_settings(cfg: dict[str, Any]) -> APISettings:
    ev=cfg.get('evaluator', {})
    vals={k: os.environ.get(str(ev.get(f'{k}_env') or d), '').strip() for k,d in [('api_key','OPENAI_API_KEY'),('base_url','OPENAI_BASE_URL'),('model','OPENAI_MODEL')]}
    if not vals['api_key'] or not vals['base_url'] or not vals['model']:
        raise RuntimeError('missing API configuration: set OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL')
    return APISettings(vals['api_key'], vals['base_url'], vals['model'], float(ev.get('temperature',0.0)), int(ev.get('max_retries',3)))

def validate_structured_response(raw: str) -> dict[str, Any]:
    try: data=json.loads(raw)
    except json.JSONDecodeError as exc: raise ValueError(f"non-JSON evaluator output: {exc.msg}") from exc
    if data.get('preferred') not in {'A','B','equal'}: raise ValueError('unknown preferred value')
    if 'confidence' not in data: raise ValueError('missing confidence')
    conf=float(data['confidence'])
    if not math.isfinite(conf) or not 0.0 <= conf <= 1.0: raise ValueError('nonfinite confidence')
    reason=data.get('reason')
    if not isinstance(reason, str) or not reason.strip(): raise ValueError('missing reason')
    blob=json.dumps(data).lower()
    if any(w in blob for w in FORBIDDEN_RESPONSE_WORDS): raise ValueError('response refers to hidden method names')
    data['confidence']=conf
    return data

def route_event(event_type: object) -> str:
    return decision_event_agent(normalize_decision_event_type(event_type))

def _features(candidate: dict[str, Any]) -> list[float]:
    vals=candidate.get('features', candidate.get('candidate_features', []))
    if isinstance(vals, dict): vals=list(vals.values())
    vals=[float(x) for x in vals]
    if not vals: raise ValueError('candidate features are missing or empty')
    if not all(math.isfinite(x) for x in vals): raise ValueError('candidate features contain non-finite values')
    return vals

def _candidate_id(c: dict[str, Any]) -> str:
    return str(c.get('candidate_id', c.get('action_id', c.get('id', c.get('action_name')))))

def feasible_pairs(state: dict[str, Any]) -> list[tuple[dict[str,Any], dict[str,Any]]]:
    cands=state.get('candidates', state.get('candidate_actions', []))
    mask=state.get('action_mask') or [bool(c.get('feasible', True)) for c in cands]
    feas=[c for c,m in zip(cands, mask) if bool(m) and c.get('feasible', True)]
    pairs=[]
    for i in range(len(feas)):
        for j in range(i+1,len(feas)):
            if _candidate_id(feas[i]) != _candidate_id(feas[j]): pairs.append((feas[i],feas[j]))
    return pairs

def normalize_decision_state(raw: dict[str, Any]) -> dict[str, Any] | None:
    try: event=normalize_decision_event_type(raw.get('event_type'))
    except Exception: return None
    agent=route_event(event)
    names=raw.get('state_feature_names') or raw.get('assignment_feature_names') or ['state_0']
    vals=raw.get('state_features') or raw.get('assignment_features') or [0.0]
    vals=[float(x) for x in vals]
    cands=raw.get('candidates') or raw.get('candidate_actions') or []
    if len(cands) < 2: return None
    return {**raw, 'agent_type':agent, 'event_type':event, 'state_feature_names':[str(x) for x in names], 'state_features':vals,
            'scenario_id':str(raw.get('scenario_id','unknown')), 'scenario_hash':str(raw.get('scenario_hash') or sha_json(raw.get('scenario_id','unknown'))),
            'episode_id':str(raw.get('episode_id','0')), 'state_id':str(raw.get('state_id', raw.get('decision_id','state'))),
            'decision_id':str(raw.get('decision_id', raw.get('state_id','state'))), 'simulation_time':float(raw.get('simulation_time', raw.get('current_time',0.0))), 'candidates':cands}

def _validate_real_observation(obs: dict[str, Any]) -> dict[str, Any] | None:
    try: event=normalize_decision_event_type(obs.get('event_type'))
    except Exception: return None
    for key in ('features','feature_names','candidate_actions','candidate_features','candidate_feature_names','action_mask'):
        if key not in obs: raise ValueError(f'missing observation field {key}')
    feats=[float(x) for x in obs['features']]
    if len(feats) != len(obs['feature_names']) or not feats or not all(math.isfinite(x) for x in feats): raise ValueError('invalid observation feature vector')
    cands=list(obs['candidate_actions']); cfeat=list(obs['candidate_features']); mask=list(obs['action_mask'])
    if not (len(cands)==len(cfeat)==len(mask)) or len(cands)<1: raise ValueError('invalid candidate dimensions')
    norm=[]
    for i,(c,cf) in enumerate(zip(cands,cfeat)):
        vals=[float(x) for x in (cf.values() if isinstance(cf,dict) else cf)]
        if len(vals) != len(obs['candidate_feature_names']) or not vals or not all(math.isfinite(x) for x in vals): raise ValueError('invalid candidate feature vector')
        payload=c if isinstance(c,dict) else {'action_id': i, 'action': c}
        norm.append({**payload, 'candidate_id': str(payload.get('candidate_id', payload.get('action_id', i))), 'action_id': int(payload.get('action_id', i)), 'features': vals, 'feature_names': [str(x) for x in obs['candidate_feature_names']], 'feasible': bool(mask[i])})
    return {'agent_type':route_event(event),'event_type':event,'state_feature_names':[str(x) for x in obs['feature_names']],'state_features':feats,'simulation_time':float(obs.get('time_min', obs.get('current_time',0.0))),'candidates':norm,'candidate_actions':norm,'candidate_feature_names':[str(x) for x in obs['candidate_feature_names']],'candidate_features':[n['features'] for n in norm],'action_mask':[bool(x) for x in mask],'observation_schema_version':OBSERVATION_SCHEMA_VERSION,'candidate_schema_version':CANDIDATE_SCHEMA_VERSION,'event_schema_version':EVENT_SCHEMA_VERSION}

def _select_rollout_action(state: dict[str,Any], rng) -> int:
    feasible=[i for i,m in enumerate(state['action_mask']) if m]
    if not feasible: raise RuntimeError('no feasible rollout actions')
    return feasible[rng.randrange(len(feasible))]

def _progress_identity(bank_hash: str, scenario_hash: str, seed: int, policy: str) -> dict[str,Any]:
    return {'bank_hash':bank_hash,'scenario_hash':scenario_hash,'collection_seed':seed,'collection_policy_id':policy,'observation_schema_version':OBSERVATION_SCHEMA_VERSION,'candidate_schema_version':CANDIDATE_SCHEMA_VERSION,'event_schema_version':EVENT_SCHEMA_VERSION}

def _collect_scenario(scenario, bank_hash: str, seed: int, policy_id: str) -> list[dict[str,Any]]:
    import random
    load_frozen_instance(scenario)
    env=DynamicDeliveryEnv(Path(scenario.instance_path))
    obs,_=env.reset(seed=seed)
    rng=random.Random(sha_json([bank_hash, scenario.scenario_id, seed, policy_id]))
    rows=[]; decision=0
    while obs.get('agent') != 'terminal':
        state=_validate_real_observation(obs)
        if state:
            state.update({'scenario_id':scenario.scenario_id,'scenario_hash':scenario.scenario_content_hash or scenario.instance_hash,'scenario_bank_hash':bank_hash,'scenario_split':scenario.split,'episode_id':scenario.scenario_id,'state_id':f'{scenario.scenario_id}:{decision}','decision_id':f'{scenario.scenario_id}:{decision}','collection_policy_id':policy_id,'collection_seed':seed})
            action=_select_rollout_action(state,rng); state['selected_rollout_action']=action; rows.append(state); decision += 1
        else:
            action=0
        obs,_,terminated,truncated,_=env.step(action)
        if terminated or truncated: break
    return rows

def _collect_from_injected(config: dict[str, Any], manifest: Path) -> list[dict[str,Any]]:
    data=json.loads(manifest.read_text()); records=[]
    for key in ('decision_states','states'):
        if isinstance(data.get(key), list): records += data[key]
    for p in data.get('decision_state_files',[]): records += read_jsonl(p)
    out=[s for r in records if (s:=normalize_decision_state(r))]
    if not out: raise RuntimeError('no decision states found in injected state manifest')
    return out

def collect_decision_states(config: dict[str, Any], *, output_root: Path|None=None, resume: bool=False) -> list[dict[str,Any]]:
    sb=config.get('scenario_bank',{}); manifest=Path(sb.get('final_train_manifest',''))
    if not manifest.is_file(): raise RuntimeError(f'formal train scenario-bank manifest missing: {manifest}')
    source_mode=sb.get('source_mode','frozen_scenario_rollout')
    if source_mode == 'injected_states': return _collect_from_injected(config, manifest)
    if source_mode != 'frozen_scenario_rollout': raise RuntimeError(f'unsupported source_mode {source_mode}')
    bank=load_scenario_bank(manifest); m=load_bank_manifest(manifest)
    if bank.split != 'train' or any(s.split != 'train' for s in bank.scenarios): raise RuntimeError('formal preference collection requires train split only')
    if not bank.bank_hash: raise RuntimeError('formal train scenario-bank bank_hash missing')
    seed=int(config.get('collection',{}).get('seed', sb.get('collection_seed', 1))); policy_id='seeded_feasible_action_sampler_v1'
    progress_root=(output_root or Path('results/formal/rlaif'))/'collection_progress'; progress_root.mkdir(parents=True,exist_ok=True)
    states=[]; counts={}
    for sc in bank.scenarios:
        ident=_progress_identity(bank.bank_hash, sc.scenario_content_hash or sc.instance_hash, seed, policy_id)
        pf=progress_root/f'{sc.scenario_id}.json'; sf=progress_root/f'{sc.scenario_id}.states.jsonl'
        if resume and pf.is_file():
            pr=json.loads(pf.read_text())
            if pr.get('identity') != ident: raise RuntimeError(f'stale collection progress for {sc.scenario_id}')
            if pr.get('completed') and sf.is_file():
                recs=read_jsonl(sf); states.extend(recs); continue
        try:
            recs=_collect_scenario(sc, bank.bank_hash, seed, policy_id); write_jsonl(sf,recs)
            ev_counts={e:sum(r['event_type']==e for r in recs) for e in DECISION_EVENT_SPECS}
            pf.write_text(json.dumps({'identity':ident,'scenario_id':sc.scenario_id,'scenario_hash':ident['scenario_hash'],'bank_hash':bank.bank_hash,'collection_seed':seed,'completed':True,'decision_counts_by_event':ev_counts,'output_state_file':str(sf),'failure_status':None,'failure_reason':None},indent=2,sort_keys=True))
            states.extend(recs)
        except Exception as exc:
            pf.write_text(json.dumps({'identity':ident,'scenario_id':sc.scenario_id,'scenario_hash':ident['scenario_hash'],'bank_hash':bank.bank_hash,'collection_seed':seed,'completed':False,'failure_status':'failed','failure_reason':str(exc)},indent=2,sort_keys=True)); raise
    for st in states: counts[st['event_type']]=counts.get(st['event_type'],0)+1
    for agent, req in REQUIRED_EVENT_COVERAGE.items():
        miss=req-set(counts)
        if miss: raise RuntimeError(f'{agent} missing event coverage {sorted(miss)}; scenarios traversed={len(bank.scenarios)}; decision counts by event={counts}')
    config['_scenario_bank_manifest_data']={'path':str(manifest),'bank_hash':bank.bank_hash,'manifest_file_hash':bank_sha256_file(manifest),'split':m.get('split'),'scenario_count':m.get('scenario_count')}
    config['_collection']={'collection_policy_id':policy_id,'collection_seed':seed}
    return states

def build_prompt(state: dict[str,Any], a: dict[str,Any], b: dict[str,Any], cfg: dict[str,Any]) -> str:
    agent=state['agent_type']; event=state['event_type']
    focus={
      'assignment':'delivery feasibility, deadline risk, expected lateness, truck capacity, bus freight capacity, locker congestion, drone feasibility, energy and downstream congestion',
      'truck':'route feasibility, parcel urgency, weight and volume capacity, travel distance and time, truck cost, downstream delivery feasibility, expected lateness',
      'bus':'BUS_TERMINAL_DEPARTURE freight loading/passenger-service implications; BUS_STATION_ARRIVAL charging duration, state of charge, passenger delay, operating delay, station load, future trip feasibility',
      'station':'parcel urgency, locker occupancy, drone availability, battery availability, charging-slot state, station power load, expected lateness, future congestion'}[agent]
    ctx={'agent_type':agent,'event_type':event,'scenario_id':state['scenario_id'],'simulation_time':state['simulation_time'],'state_features':dict(zip(state['state_feature_names'],state['state_features'])),'candidate_A':a,'candidate_B':b,'consequence_A':a.get('consequence',{}),'consequence_B':b.get('consequence',{})}
    return 'Compare candidate A and B for the active operational decision. Consider: '+focus+f". Event type is {event}. Return only JSON with preferred (A/B/equal), confidence, criteria, reason. Do not mention learning algorithms. Context: "+json.dumps(ctx,sort_keys=True)

def cache_key(state:dict[str,Any], a:dict[str,Any], b:dict[str,Any], settings:APISettings, cfg:dict[str,Any]) -> str:
    return sha_json({'agent_type':state['agent_type'],'event_type':state['event_type'],'scenario_hash':state['scenario_hash'],'decision_state_hash':sha_json({'id':state['state_id'],'features':state['state_features']}),'candidate_pair_hash':sha_json(sorted([_candidate_id(a),_candidate_id(b)])),'prompt_version':cfg['evaluator'].get('prompt_version',PROMPT_VERSION),'response_schema_version':cfg['evaluator'].get('structured_output_schema_version',RESPONSE_SCHEMA_VERSION),'evaluator_model':settings.model_name,'temperature':settings.temperature})

def make_preference_record(state:dict[str,Any], a:dict[str,Any], b:dict[str,Any], response:dict[str,Any], split:str, settings:APISettings, cfg:dict[str,Any]) -> dict[str,Any]:
    aid,bid=_candidate_id(a),_candidate_id(b); outcome={'A':'candidate_a','B':'candidate_b','equal':'tie'}[response['preferred']]
    cf_names=[str(x) for x in a.get('feature_names') or b.get('feature_names') or [f'f{i}' for i in range(len(_features(a)))]]
    pair_hash=sha_json([state['scenario_hash'], state['state_id'], sorted([aid,bid])]); state_hash=sha_json({'scenario_hash':state['scenario_hash'],'id':state['state_id'],'features':state['state_features']})
    return {'preference_id':sha_json([state['state_id'],aid,bid,settings.model_name]),'agent_type':state['agent_type'],'event_type':state['event_type'],'scenario_id':state['scenario_id'],'scenario_hash':state['scenario_hash'],'scenario_bank_hash':state.get('scenario_bank_hash'),'scenario_split':state.get('scenario_split','train'),'episode_id':state['episode_id'],'state_id':state['state_id'],'decision_id':state['decision_id'],'simulation_time':state['simulation_time'],'state_feature_schema_version':str(OBSERVATION_SCHEMA_VERSION),'state_feature_names':state['state_feature_names'],'state_features':state['state_features'],'candidate_a_id':aid,'candidate_b_id':bid,'original_candidate_a_id':aid,'original_candidate_b_id':bid,'displayed_first_candidate_id':aid,'displayed_second_candidate_id':bid,'candidate_a_feature_names':cf_names,'candidate_b_feature_names':cf_names,'candidate_a_features':_features(a),'candidate_b_features':_features(b),'candidate_a_id_features':a,'candidate_b_id_features':b,'candidate_a_consequence':a.get('consequence',{}),'candidate_b_consequence':b.get('consequence',{}),'action_mask':state.get('action_mask',[True]*len(state['candidates'])),'prompt_version':cfg['evaluator'].get('prompt_version',PROMPT_VERSION),'evaluator_prompt_version':cfg['evaluator'].get('prompt_version',PROMPT_VERSION),'response_schema_version':cfg['evaluator'].get('structured_output_schema_version',RESPONSE_SCHEMA_VERSION),'dataset_split':split,'state_hash':state_hash,'candidate_pair_hash':pair_hash,'reversed_candidate_pair_hash':pair_hash,'original_outcome':outcome,'label_source':'external_evaluator_api','evaluator_model':settings.model_name,'confidence':response['confidence'],'criteria':response.get('criteria',{}),'reason':response['reason'],'observation_schema_version':OBSERVATION_SCHEMA_VERSION,'candidate_schema_version':CANDIDATE_SCHEMA_VERSION,'event_schema_version':EVENT_SCHEMA_VERSION,'collection_policy_id':state.get('collection_policy_id'),'collection_seed':state.get('collection_seed'),'selected_rollout_action':state.get('selected_rollout_action'),'created_at':datetime.now(timezone.utc).isoformat()}

def validate_split_isolation(rows:list[dict[str,Any]])->None:
    by={k:{} for k in ('scenario_id','scenario_hash','decision_id','state_hash','candidate_pair_hash','reversed_candidate_pair_hash')}
    for r in rows:
        sp=r['dataset_split']
        for k,m in by.items():
            v=r[k]
            if v in m and m[v]!=sp: raise ValueError(f'split leakage detected for {k}={v}')
            m[v]=sp

def generate(config_path:Path, output_root:Path, *, resume:bool=False, api_call:Callable[[str,APISettings],str]|None=None) -> dict[str,Any]:
    cfg=load_yaml(config_path); settings=evaluator_settings(cfg)
    states=collect_decision_states(cfg, output_root=output_root, resume=resume)
    split_cfg=cfg.get('preference_split',{}); split=grouped_split(states, split_cfg.get('train_fraction',.7), split_cfg.get('validation_fraction',.15), split_cfg.get('test_fraction',.15), split_cfg.get('seed',1), 'scenario')
    split_by_state={s['state_id']:name for name,rs in split['records'].items() for s in rs}
    prefs_dir=output_root/'preferences'; failed_dir=output_root/'failed'; cache_dir=Path(cfg.get('evaluator',{}).get('cache_dir') or output_root/'evaluator_cache')
    failed_dir.mkdir(parents=True,exist_ok=True); cache_dir.mkdir(parents=True,exist_ok=True)
    call=api_call or _default_api_call; by_agent={a:[] for a in AGENT_TYPES}; failures={a:[] for a in AGENT_TYPES}
    for st in states:
        for a,b in feasible_pairs(st):
            ck=cache_key(st,a,b,settings,cfg); cp=cache_dir/f'{ck}.json'; raw=''; err=''
            try:
                if resume and cp.is_file(): resp=validate_structured_response(json.loads(cp.read_text())['raw_response'])
                else:
                    for attempt in range(settings.max_retries):
                        try: raw=call(build_prompt(st,a,b,cfg),settings); resp=validate_structured_response(raw); cp.write_text(json.dumps({'identity':ck,'raw_response':raw},sort_keys=True)); break
                        except Exception as exc:
                            err=str(exc); time.sleep(0.01*(attempt+1))
                    else: raise ValueError(err)
                by_agent[st['agent_type']].append(make_preference_record(st,a,b,resp,split_by_state[st['state_id']],settings,cfg))
            except Exception as exc:
                failures[st['agent_type']].append({'agent_type':st['agent_type'],'event_type':st['event_type'],'state_id':st['state_id'],'error':str(exc),'raw_response':raw})
    manifest={'status':'complete','agents':{},'split_hash':split['hash'],'scenario_bank_manifest':str(cfg['scenario_bank']['final_train_manifest']),'scenario_bank':cfg.get('_scenario_bank_manifest_data',{}),'collection':cfg.get('_collection',{})}
    for agent,rows in by_agent.items():
        validate_split_isolation(rows)
        if REQUIRED_EVENT_COVERAGE[agent] - {r['event_type'] for r in rows}: raise RuntimeError(f'{agent} missing event coverage')
        target=int(cfg['agents'][agent].get('target_valid_pair_count',0))
        if len([r for r in rows if r['original_outcome'] in {'candidate_a','candidate_b'}]) < target: raise RuntimeError(f'{agent} usable-label target not met')
        out=Path(cfg['agents'][agent]['output_preferences']);
        if not out.is_absolute(): out=prefs_dir/f'preference_{agent}.jsonl'
        write_jsonl(out,rows); write_jsonl(failed_dir/f'{agent}_failed.jsonl', failures[agent])
        manifest['agents'][agent]={'path':str(out),'count':len(rows),'counts_by_split':{s:sum(r['dataset_split']==s for r in rows) for s in ('train','validation','test')},'hash':sha_file(out),'events':sorted({r['event_type'] for r in rows})}
    (output_root/'preference_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
    return manifest

def main(argv=None)->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--output-root',type=Path,default=Path('results/formal/rlaif')); ap.add_argument('--resume',action='store_true')
    ns=ap.parse_args(argv)
    try: print(json.dumps(generate(ns.config,ns.output_root,resume=ns.resume),indent=2,sort_keys=True)); return 0
    except Exception as exc: print(f'formal preference generation failed: {exc}',file=sys.stderr); return 2
if __name__=='__main__': raise SystemExit(main())
