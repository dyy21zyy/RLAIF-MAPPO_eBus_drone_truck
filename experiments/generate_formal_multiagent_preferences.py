"""Formal four-agent RLAIF preference generation.

Collects normalized decision records, builds feasible same-state candidate pairs,
queries the configured OpenAI-compatible evaluator, validates JSON responses, and
writes one canonical preference JSONL per RLAIF agent. Runtime never fabricates
labels; tests may inject a deterministic evaluator callable.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from rlaif.ai_evaluator import APISettings, _default_api_call
from rlaif.grouped_split import grouped_split
from rlaif.preference_dataset import write_jsonl, read_jsonl
from rlaif.preference_quality import (ASSIGNMENT_PROMPT_VERSION, ASSIGNMENT_RESPONSE_SCHEMA_VERSION,
    CONSEQUENCE_MODE, QUALITY_GATE_VERSION, assignment_pair_type, validate_assignment_v2)
from envs import DynamicDeliveryEnv
from envs.reward_components import REWARD_COMPONENTS
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
    ev = cfg.get("evaluator", {})

    vals = {
        key: os.environ.get(
            str(ev.get(f"{key}_env") or default),
            "",
        ).strip()
        for key, default in (
            ("api_key", "OPENAI_API_KEY"),
            ("base_url", "OPENAI_BASE_URL"),
            ("model", "OPENAI_MODEL"),
        )
    }

    if not vals["api_key"] or not vals["base_url"] or not vals["model"]:
        raise RuntimeError(
            "missing API configuration: set OPENAI_API_KEY, "
            "OPENAI_BASE_URL, and OPENAI_MODEL"
        )

    enable_thinking = ev.get("enable_thinking", False)

    if not isinstance(enable_thinking, bool):
        raise ValueError("evaluator.enable_thinking must be boolean")

    if enable_thinking is not False:
        raise ValueError(
            "formal assignment evaluation requires enable_thinking=false"
        )

    timeout_seconds = float(ev.get("timeout_seconds", 60))

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError(
            "evaluator.timeout_seconds must be positive and finite"
        )

    max_retries = int(ev.get("max_retries", 3))

    if max_retries < 1:
        raise ValueError("evaluator.max_retries must be at least 1")

    return APISettings(
        api_key=vals["api_key"],
        api_base_url=vals["base_url"],
        model_name=vals["model"],
        temperature=float(ev.get("temperature", 0.0)),
        max_retries=max_retries,
        enable_thinking=enable_thinking,
        timeout_seconds=timeout_seconds,
    )


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

def _assignment_v2_enabled(cfg: dict[str, Any], agent: str) -> bool:
    return agent == 'assignment' and cfg.get('evaluator',{}).get('assignment_structured_output_schema_version') == ASSIGNMENT_RESPONSE_SCHEMA_VERSION

def resolve_versions(cfg: dict[str, Any], agent: str) -> tuple[str,str]:
    ev=cfg.get('evaluator',{})
    if _assignment_v2_enabled(cfg,agent):
        return ev.get('assignment_prompt_version',ASSIGNMENT_PROMPT_VERSION), ev.get('assignment_structured_output_schema_version',ASSIGNMENT_RESPONSE_SCHEMA_VERSION)
    return ev.get('prompt_version',PROMPT_VERSION), ev.get('structured_output_schema_version',RESPONSE_SCHEMA_VERSION)

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


def prepare_preference_collection_environment(instance_path: str | Path) -> DynamicDeliveryEnv:
    """Load an instance for raw, unscaled preference-trajectory collection."""
    # Preference collection needs state/action/consequence trajectories, not normalized
    # rewards. Phase 3 separately estimates final train-bank scales for MAPPO configs.
    instance_path = Path(instance_path)
    manifest_path = instance_path / "instance.json" if instance_path.is_dir() else instance_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime_config = json.loads(json.dumps(manifest.get("config_snapshot")))
    runtime_config.setdefault("reward", {})["apply_reference_scales"] = False
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as config_file:
        json.dump(runtime_config, config_file)
        config_file.flush()
        env = DynamicDeliveryEnv(instance_path, config_path=config_file.name)
    env.config.setdefault("reward", {})["apply_reference_scales"] = False
    env.reward_reference_scales = {component: 1.0 for component in REWARD_COMPONENTS}
    return env

def parse_agent_scope(value: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Return a canonical, non-empty agent scope (the CLI accepts comma/space lists)."""
    if value is None:
        return tuple(AGENT_TYPES)
    raw = value if isinstance(value, (list, tuple)) else value.replace(",", " ").split()
    scope = tuple(dict.fromkeys(str(item).strip().lower() for item in raw if str(item).strip()))
    unknown = sorted(set(scope) - set(AGENT_TYPES))
    if not scope:
        raise ValueError("selected agent scope must not be empty")
    if unknown:
        raise ValueError(f"unknown agent(s) {unknown}; expected one or more of {list(AGENT_TYPES)}")
    return tuple(agent for agent in AGENT_TYPES if agent in scope)


def _progress_identity(bank_hash: str, scenario_hash: str, seed: int, policy: str,
                       selected_agents: tuple[str, ...] = tuple(AGENT_TYPES)) -> dict[str,Any]:
    return {'bank_hash':bank_hash,'scenario_hash':scenario_hash,'collection_seed':seed,'collection_policy_id':policy,'selected_agents':list(selected_agents),'observation_schema_version':OBSERVATION_SCHEMA_VERSION,'candidate_schema_version':CANDIDATE_SCHEMA_VERSION,'event_schema_version':EVENT_SCHEMA_VERSION}

def _collect_scenario(scenario, bank_hash: str, seed: int, policy_id: str) -> list[dict[str,Any]]:
    import random
    load_frozen_instance(scenario)
    env=prepare_preference_collection_environment(Path(scenario.instance_path))
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

def collect_decision_states(config: dict[str, Any], *, output_root: Path|None=None, resume: bool=False,
                            selected_agents: tuple[str, ...] = tuple(AGENT_TYPES)) -> list[dict[str,Any]]:
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
        ident=_progress_identity(bank.bank_hash, sc.scenario_content_hash or sc.instance_hash, seed, policy_id, selected_agents)
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
    for agent in selected_agents:
        req = REQUIRED_EVENT_COVERAGE[agent]
        miss=req-set(counts)
        if miss: raise RuntimeError(f'{agent} missing event coverage {sorted(miss)}; scenarios traversed={len(bank.scenarios)}; decision counts by event={counts}')
    config['_scenario_bank_manifest_data']={'path':str(manifest),'bank_hash':bank.bank_hash,'manifest_file_hash':bank_sha256_file(manifest),'split':m.get('split'),'scenario_count':m.get('scenario_count')}
    config['_collection']={'collection_policy_id':policy_id,'collection_seed':seed}
    return states

def build_prompt(
    state: dict[str, Any],
    a: dict[str, Any],
    b: dict[str, Any],
    cfg: dict[str, Any],
) -> str:
    agent = state["agent_type"]
    event = state["event_type"]

    focus = {
        "assignment": (
            "delivery feasibility, delivery time, expected lateness, "
            "truck distance/time, mode-applicable bus wait/linehaul, "
            "drone time, locker congestion, and station power margin"
        ),
        "truck": (
            "route feasibility, parcel urgency, weight and volume "
            "capacity, travel distance and time, truck cost, "
            "downstream delivery feasibility, expected lateness"
        ),
        "bus": (
            "BUS_TERMINAL_DEPARTURE freight loading/passenger-service "
            "implications; BUS_STATION_ARRIVAL charging duration, "
            "state of charge, passenger delay, operating delay, "
            "station load, future trip feasibility"
        ),
        "station": (
            "parcel urgency, locker occupancy, drone availability, "
            "battery availability, charging-slot state, station power "
            "load, expected lateness, future congestion"
        ),
    }[agent]

    context = {
        "agent_type": agent,
        "event_type": event,
        "scenario_id": state["scenario_id"],
        "simulation_time": state["simulation_time"],
        "state_features": dict(
            zip(
                state["state_feature_names"],
                state["state_features"],
            )
        ),
        "candidate_A": a,
        "candidate_B": b,
        "consequence_A": a.get("consequence", {}),
        "consequence_B": b.get("consequence", {}),
    }

    if _assignment_v2_enabled(cfg, agent):
        canonical = [
            "delivery_feasibility",
            "delivery_time",
            "expected_lateness",
            "deadline_risk",
            "truck_distance",
            "truck_time",
            "truck_capacity",
            "bus_wait_time",
            "bus_linehaul_time",
            "bus_freight_capacity",
            "drone_time",
            "drone_feasibility",
            "locker_congestion",
            "station_power_margin",
            "downstream_congestion",
        ]

        schema = (
            "Return only JSON with preferred (A/B/equal), numeric "
            "confidence, a nonempty criteria list, a nonempty evidence "
            "list of {metric, better_candidate}, and reason. "
            "criteria must contain only these exact canonical names: "
            + ", ".join(canonical)
            + ". Candidate feature names must not be copied into criteria. "
            "Evidence metrics must refer to observable candidate metrics. "
            "TLD does not use a bus. "
            "TD does not use a locker or drone. "
            "Do not claim energy, emissions, fuel consumption, or "
            "simulated downstream consequences unless directly provided."
        )
    else:
        schema = (
            "Return only JSON with preferred (A/B/equal), confidence, "
            "criteria, reason."
        )

    consequence = (
        "Candidate payloads are estimated candidate attributes, not "
        "simulated downstream consequences. "
        if agent == "assignment"
        else ""
    )

    return (
        "Compare candidate A and B for the active operational decision. "
        + consequence
        + "Consider: "
        + focus
        + f". Event type is {event}. "
        + schema
        + " Do not mention learning algorithms. Context: "
        + json.dumps(context, sort_keys=True)
    )


def cache_key(state:dict[str,Any], a:dict[str,Any], b:dict[str,Any], settings:APISettings, cfg:dict[str,Any], selected_agents: tuple[str,...] = tuple(AGENT_TYPES)) -> str:
    prompt_version,response_version=resolve_versions(cfg,state['agent_type']); ev=cfg.get('evaluator',{})
    return sha_json({'agent_type':state['agent_type'],'event_type':state['event_type'],'scenario_hash':state['scenario_hash'],'decision_state_hash':sha_json({'id':state['state_id'],'features':state['state_features']}),'candidate_pair_hash':sha_json(sorted([_candidate_id(a),_candidate_id(b)])),'prompt_version':prompt_version,'response_schema_version':response_version,'quality_gate_version':ev.get('assignment_quality_gate_version') if state['agent_type']=='assignment' else None,'consequence_mode':ev.get('assignment_consequence_mode') if state['agent_type']=='assignment' else None,'evaluator_model':settings.model_name,'temperature':settings.temperature,'selected_agents':list(selected_agents),'observation_schema_version':OBSERVATION_SCHEMA_VERSION,'candidate_schema_version':CANDIDATE_SCHEMA_VERSION,'event_schema_version':EVENT_SCHEMA_VERSION})

def make_preference_record(state:dict[str,Any], a:dict[str,Any], b:dict[str,Any], response:dict[str,Any], split:str, settings:APISettings, cfg:dict[str,Any]) -> dict[str,Any]:
    aid,bid=_candidate_id(a),_candidate_id(b); outcome={'A':'candidate_a','B':'candidate_b','equal':'tie'}[response['preferred']]
    cf_names=[str(x) for x in a.get('feature_names') or b.get('feature_names') or [f'f{i}' for i in range(len(_features(a)))]]
    pair_hash=sha_json([state['scenario_hash'], state['state_id'], sorted([aid,bid])]); state_hash=sha_json({'scenario_hash':state['scenario_hash'],'id':state['state_id'],'features':state['state_features']})
    pv,rv=resolve_versions(cfg,state['agent_type'])
    return {'preference_id':sha_json([state['state_id'],aid,bid,settings.model_name]),'agent_type':state['agent_type'],'event_type':state['event_type'],'scenario_id':state['scenario_id'],'scenario_hash':state['scenario_hash'],'scenario_bank_hash':state.get('scenario_bank_hash'),'scenario_split':state.get('scenario_split','train'),'episode_id':state['episode_id'],'state_id':state['state_id'],'decision_id':state['decision_id'],'simulation_time':state['simulation_time'],'state_feature_schema_version':str(OBSERVATION_SCHEMA_VERSION),'state_feature_names':state['state_feature_names'],'state_features':state['state_features'],'candidate_a_id':aid,'candidate_b_id':bid,'original_candidate_a_id':aid,'original_candidate_b_id':bid,'displayed_first_candidate_id':aid,'displayed_second_candidate_id':bid,'candidate_a_feature_names':cf_names,'candidate_b_feature_names':cf_names,'candidate_a_features':_features(a),'candidate_b_features':_features(b),'candidate_a_id_features':a,'candidate_b_id_features':b,'candidate_a_consequence':a.get('consequence',{}),'candidate_b_consequence':b.get('consequence',{}),'action_mask':state.get('action_mask',[True]*len(state['candidates'])),'prompt_version':pv,'evaluator_prompt_version':pv,'response_schema_version':rv,'quality_gate_version':response.get('quality_gate_version'),'consequence_mode':response.get('consequence_mode'),'dataset_split':split,'state_hash':state_hash,'candidate_pair_hash':pair_hash,'reversed_candidate_pair_hash':pair_hash,'original_outcome':outcome,'label_source':'external_evaluator_api','evaluator_model':settings.model_name,'evaluator_enable_thinking':settings.enable_thinking,'source_generation_enable_thinking':response.get('_source_generation_enable_thinking',settings.enable_thinking),'confidence':response['confidence'],'criteria':response.get('criteria',{}),'evidence':response.get('validated_evidence',response.get('evidence')),'raw_evaluator_reason':response.get('raw_evaluator_reason',response['reason']),'reason':response['reason'],'observation_schema_version':OBSERVATION_SCHEMA_VERSION,'candidate_schema_version':CANDIDATE_SCHEMA_VERSION,'event_schema_version':EVENT_SCHEMA_VERSION,'collection_policy_id':state.get('collection_policy_id'),'collection_seed':state.get('collection_seed'),'selected_rollout_action':state.get('selected_rollout_action'),'created_at':datetime.now(timezone.utc).isoformat()}

def validate_split_isolation(rows:list[dict[str,Any]])->None:
    by={k:{} for k in ('scenario_id','scenario_hash','decision_id','state_hash','candidate_pair_hash','reversed_candidate_pair_hash')}
    for r in rows:
        sp=r['dataset_split']
        for k,m in by.items():
            v=r[k]
            if v in m and m[v]!=sp: raise ValueError(f'split leakage detected for {k}={v}')
            m[v]=sp


def _pair_identity(state:dict[str,Any], a:dict[str,Any], b:dict[str,Any]) -> str:
    return sha_json([state['scenario_hash'], state['state_id'], sorted([_candidate_id(a), _candidate_id(b)])])

def _stratum_key(state:dict[str,Any], split:str) -> tuple[str,str,str]:
    return (state['agent_type'], state['event_type'], split)

def _required_strata(agent:str, cfg:dict[str,Any]) -> set[tuple[str,str,str]]:
    events=set(cfg.get('agents',{}).get(agent,{}).get('supported_event_types') or REQUIRED_EVENT_COVERAGE[agent])
    return {(agent,e,s) for e in events for s in ('train','validation','test')}

def _build_pair_pool(states:list[dict[str,Any]], split_by_state:dict[str,str], seed:int) -> tuple[list[dict[str,Any]], dict[str,Any]]:
    pool=[]; counts={}
    for idx,st in enumerate(states):
        split=split_by_state[st['state_id']]
        pairs=feasible_pairs(st)
        key=_stratum_key(st,split); counts[key]=counts.get(key,0)+len(pairs)
        for pi,(a,b) in enumerate(pairs):
            ident=_pair_identity(st,a,b)
            pool.append({'state':st,'a':a,'b':b,'state_index':idx,'pair_index':pi,'pair_identity':ident,'stratum':key,
                         'order_key':sha_json([seed, st['agent_type'], st['event_type'], split, st['scenario_hash'], st['state_id'], ident])})
    return pool, counts

def _select_agent_pairs(agent:str, pool:list[dict[str,Any]], budget:int, seed:int) -> list[dict[str,Any]]:
    agent_pool=[p for p in pool if p['state']['agent_type']==agent]
    by_stratum={}
    for p in sorted(agent_pool, key=lambda x:x['order_key']): by_stratum.setdefault(p['stratum'],[]).append(p)
    strata=sorted(by_stratum, key=lambda k: sha_json([seed,*k]))
    chosen=[]; chosen_ids=set(); used_states=set()
    def rounds(first_pass:bool):
        made=True
        while made and len(chosen)<budget:
            made=False
            for sk in strata:
                for p in by_stratum[sk]:
                    if p['pair_identity'] in chosen_ids: continue
                    sid=p['state']['state_id']
                    if first_pass and sid in used_states: continue
                    chosen.append(p); chosen_ids.add(p['pair_identity']); used_states.add(sid); made=True; break
                if len(chosen)>=budget: break
    rounds(True); rounds(False)
    return chosen[:budget]

def _count_records(rows:list[dict[str,Any]]) -> dict[str,Any]:
    return {'by_event':{e:sum(r['event_type']==e for r in rows) for e in sorted({r['event_type'] for r in rows})},
            'by_split':{s:sum(r.get('dataset_split')==s for r in rows) for s in ('train','validation','test')},
            'by_event_and_split':{f'{e}|{s}':sum(r['event_type']==e and r.get('dataset_split')==s for r in rows) for e in sorted({r['event_type'] for r in rows}) for s in ('train','validation','test')}}

def _coverage_satisfied(agent:str, rows:list[dict[str,Any]], cfg:dict[str,Any]) -> bool:
    binary=[r for r in rows if r.get('original_outcome') in {'candidate_a','candidate_b'}]
    have={(r['agent_type'],r['event_type'],r.get('dataset_split')) for r in binary}
    return _required_strata(agent,cfg) <= have

def generate(config_path:Path, output_root:Path, *, resume:bool=False, dry_run:bool=False, api_call:Callable[[str,APISettings],str]|None=None, agents: str | list[str] | tuple[str,...] | None=None, cache_dir: Path | None=None) -> dict[str,Any]:
    cfg=load_yaml(config_path); settings=None if dry_run else evaluator_settings(cfg)
    selected_agents=parse_agent_scope(agents)
    states=collect_decision_states(cfg, output_root=output_root, resume=resume, selected_agents=selected_agents)
    split_cfg=cfg.get('preference_split',{}); split=grouped_split(states, split_cfg.get('train_fraction',.7), split_cfg.get('validation_fraction',.15), split_cfg.get('test_fraction',.15), split_cfg.get('seed',1), 'scenario')
    split_by_state={s['state_id']:name for name,rs in split['records'].items() for s in rs}
    seed=int(split_cfg.get('seed',1))
    pool, pool_counts=_build_pair_pool(states, split_by_state, seed)
    prefs_dir=output_root/'preferences'; failed_dir=output_root/'failed'; cache_dir=cache_dir or Path(cfg.get('evaluator',{}).get('cache_dir') or output_root/'evaluator_cache')
    if not dry_run:
        failed_dir.mkdir(parents=True,exist_ok=True); cache_dir.mkdir(parents=True,exist_ok=True)
    manifest={'status':'dry_run' if dry_run else 'complete','selected_agents':list(selected_agents),'agent_scope_hash':sha_json(list(selected_agents)),'agents':{},'split_hash':split['hash'],'scenario_bank_manifest':str(cfg['scenario_bank']['final_train_manifest']),'scenario_bank':cfg.get('_scenario_bank_manifest_data',{}),'collection':cfg.get('_collection',{}),'cache_directory':str(cache_dir)}
    default_budget={'assignment':600,'truck':480,'bus':600,'station':480}
    report_lines=[]
    call=api_call or _default_api_call
    for agent in selected_agents:
        acfg=cfg.get('agents',{}).get(agent,{})
        target=int(acfg.get('target_valid_pair_count',0)); budget=int(acfg.get('max_api_attempts', default_budget[agent]))
        selected=_select_agent_pairs(agent,pool,budget,seed)
        agent_pool=[p for p in pool if p['state']['agent_type']==agent]
        selected_counts={}
        for p in selected: selected_counts[p['stratum']]=selected_counts.get(p['stratum'],0)+1
        pool_total=len(agent_pool)
        if dry_run:
            manifest['agents'][agent]={'candidate_pair_pool_count':pool_total,'selected_pair_count':len(selected),'external_api_call_count':0,'cache_hit_count':0,'valid_binary_label_count':0,'tie_count':0,'failure_count':0,'target_valid_pair_count':target,'max_api_attempts':budget,'pool_counts_by_event_split':{f'{a}|{e}|{s}':c for (a,e,s),c in sorted(pool_counts.items()) if a==agent},'selected_counts_by_event_split':{f'{a}|{e}|{s}':c for (a,e,s),c in sorted(selected_counts.items()) if a==agent}}
            report_lines.append(f"{agent}: pool={pool_total} selected_budget={len(selected)} target={target} max_api_attempts={budget}")
            continue
        out=Path(acfg['output_preferences']);
        if not out.is_absolute(): out=prefs_dir/f'preference_{agent}.jsonl'
        existing=read_jsonl(out) if resume and out.is_file() else []
        rows=list(existing); seen={r.get('preference_id') for r in rows}; failures=[]; external_calls=0; cache_hits=0
        for p in selected:
            if len([r for r in rows if r.get('original_outcome') in {'candidate_a','candidate_b'}]) >= target and _coverage_satisfied(agent,rows,cfg): break
            st,a,b=p['state'],p['a'],p['b']
            ck=cache_key(st,a,b,settings,cfg,selected_agents); cp=cache_dir/f'{ck}.json'; raw=''; err=''
            try:
                if cp.is_file():
                    cache_payload = json.loads(cp.read_text())
                    raw = cache_payload["raw_response"]
                    parsed = validate_structured_response(raw)
                    resp = (
                        validate_assignment_v2(
                            parsed,
                            a,
                            b,
                            tolerance=float(
                                cfg.get("evaluator", {}).get(
                                    "metric_comparison_tolerance",
                                    1e-9,
                                )
                            ),
                        )
                        if _assignment_v2_enabled(cfg, agent)
                        else parsed
                    )
                    resp["_source_generation_enable_thinking"] = (
                        cache_payload.get(
                            "enable_thinking",
                            "legacy_unspecified",
                        )
                    )
                    cache_hits += 1
                else:
                    if external_calls >= budget: raise RuntimeError(f'{agent} maximum API-attempt budget exhausted ({budget}) before target was met')
                    for attempt in range(settings.max_retries):
                        if external_calls >= budget: raise RuntimeError(f'{agent} maximum API-attempt budget exhausted ({budget}) before target was met')
                        try:
                            raw=call(build_prompt(st,a,b,cfg),settings); external_calls+=1
                            parsed=validate_structured_response(raw)
                            resp = (
                                validate_assignment_v2(
                                    parsed,
                                    a,
                                    b,
                                    tolerance=float(
                                        cfg.get("evaluator", {}).get(
                                            "metric_comparison_tolerance",
                                            1e-9,
                                        )
                                    ),
                                )
                                if _assignment_v2_enabled(cfg, agent)
                                else parsed
                            )
                            resp["_source_generation_enable_thinking"] = (
                                settings.enable_thinking
                            )
                            cp.write_text(
                                json.dumps(
                                    {
                                        "identity": ck,
                                        "raw_response": raw,
                                        "prompt_version": resolve_versions(
                                            cfg,
                                            agent,
                                        )[0],
                                        "response_schema_version": resolve_versions(
                                            cfg,
                                            agent,
                                        )[1],
                                        "quality_gate_version": (
                                            QUALITY_GATE_VERSION
                                            if agent == "assignment"
                                            else None
                                        ),
                                        "consequence_mode": (
                                            CONSEQUENCE_MODE
                                            if agent == "assignment"
                                            else None
                                        ),
                                        "enable_thinking": (
                                            settings.enable_thinking
                                        ),
                                        "timeout_seconds": (
                                            settings.timeout_seconds
                                        ),
                                        "evaluator_model": (
                                            settings.model_name
                                        ),
                                        "validation_status": "passed",
                                    },
                                    sort_keys=True,
                                )
                            )
                            break
                        except Exception as exc:
                            err=str(exc); time.sleep(0.01*(attempt+1))
                    else: raise ValueError(err)
                rec=make_preference_record(st,a,b,resp,split_by_state[st['state_id']],settings,cfg)
                if rec['preference_id'] not in seen: rows.append(rec); seen.add(rec['preference_id'])
            except Exception as exc:
                failures.append({'agent_type':agent,'event_type':st['event_type'],'state_id':st['state_id'],'candidate_pair_hash':p['pair_identity'],'error':str(exc),'raw_response':raw})
        validate_split_isolation(rows)
        valid_binary=len([r for r in rows if r['original_outcome'] in {'candidate_a','candidate_b'}])
        if valid_binary < target: raise RuntimeError(f'{agent} usable-label target {target} not met within API-attempt budget {budget}; valid_binary={valid_binary}')
        if not _coverage_satisfied(agent,rows,cfg): raise RuntimeError(f'{agent} missing required event/split coverage after selection')
        write_jsonl(out,rows); write_jsonl(failed_dir/f'{agent}_failed.jsonl', failures)
        counts=_count_records(rows)
        manifest['agents'][agent]={'path':str(out),'count':len(rows),'counts_by_split':counts['by_split'],'hash':sha_file(out),'events':sorted({r['event_type'] for r in rows}),'candidate_pair_pool_count':pool_total,'selected_pair_count':len(selected),'external_api_call_count':external_calls,'cache_hit_count':cache_hits,'valid_binary_label_count':valid_binary,'tie_count':sum(r['original_outcome']=='tie' for r in rows),'failure_count':len(failures),'target_valid_pair_count':target,'max_api_attempts':budget,'counts_by_event':counts['by_event'],'counts_by_event_and_split':counts['by_event_and_split'],'pool_counts_by_event_split':{f'{a}|{e}|{s}':c for (a,e,s),c in sorted(pool_counts.items()) if a==agent},'selected_counts_by_event_split':{f'{a}|{e}|{s}':c for (a,e,s),c in sorted(selected_counts.items()) if a==agent}}
    (output_root/'preference_manifest.json').parent.mkdir(parents=True,exist_ok=True)
    (output_root/'preference_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
    if dry_run: print('\n'.join(report_lines))
    return manifest

def main(argv=None)->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--output-root',type=Path,default=Path('results/formal/rlaif')); ap.add_argument('--agents',nargs='+'); ap.add_argument('--cache-dir',type=Path); ap.add_argument('--resume',action='store_true'); ap.add_argument('--dry-run',action='store_true')
    ns=ap.parse_args(argv)
    try: print(json.dumps(generate(ns.config,ns.output_root,resume=ns.resume,dry_run=ns.dry_run,agents=ns.agents,cache_dir=ns.cache_dir),indent=2,sort_keys=True)); return 0
    except Exception as exc: print(f'formal preference generation failed: {exc}',file=sys.stderr); return 2
if __name__=='__main__': raise SystemExit(main())
