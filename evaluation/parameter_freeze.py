"""Canonical final-experiment parameter freeze validation and hashing."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import copy, hashlib, json, subprocess
import yaml

CATEGORIES={"scientific_fixed","estimated_and_frozen","method_specific","seed_specific","runtime_derived","path_only"}
STATUSES={"PARAMETER_FREEZE_TEMPLATE_VALID","PARAMETER_FREEZE_READY","BLOCKED_SCENARIO_BANK_HASH","BLOCKED_REWARD_SCALE_HASH","BLOCKED_REWARD_CHECKPOINT_HASH","BLOCKED_METHOD_CONFIG_DIFFERENCE","BLOCKED_UNRESOLVED_PARAMETER","BLOCKED_POLICY_CONFIGURATION"}
CANONICAL_AGENTS=("assignment","truck","bus","station")
DECISION_EVENT_TO_AGENT={"PARCEL_RELEASE":"assignment","TRUCK_AVAILABLE":"truck","BUS_TERMINAL_DEPARTURE":"bus","BUS_STATION_ARRIVAL":"bus","STATION_OPERATION":"station"}
STATION_BASELINE_ACTIONS=("dispatch_drone","idle")
CANONICAL_REWARD_COMPONENTS=("passenger_delay","bus_operating_delay","parcel_lateness","energy_cost","power_overload","bus_battery_violation","locker_overflow","truck_cost","undelivered","battery_shortage","infeasible_action")
PLACEHOLDER_PREFIXES=("REPLACE_WITH_FINAL_","REPLACE_WITH_REAL_","PLACEHOLDER")
BLOCKED_PLACEHOLDER_STATUS={"bank_hash":"BLOCKED_SCENARIO_BANK_HASH","scale":"BLOCKED_REWARD_SCALE_HASH","reward":"BLOCKED_REWARD_CHECKPOINT_HASH","policy":"BLOCKED_POLICY_CONFIGURATION"}

@dataclass(frozen=True)
class FrozenParameter:
    key: str
    value: Any
    category: str
    source: str
    rationale: str
    allowed_method_differences: tuple[str,...]=()

class ParameterFreezeError(ValueError): pass
class DuplicateFrozenParameterError(ParameterFreezeError): pass
class UnresolvedParameterFreezeError(ParameterFreezeError): pass

def load_freeze_template(path: str|Path)->dict[str,Any]:
    return yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}

def _is_placeholder(v:Any)->bool:
    return isinstance(v,str) and any(v.startswith(p) for p in PLACEHOLDER_PREFIXES)

def _walk(node:Any, prefix:str=""):
    if isinstance(node,dict):
        yield prefix,node
        for k,v in node.items():
            yield from _walk(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(node,list):
        for i,v in enumerate(node): yield from _walk(v, f"{prefix}[{i}]")
    else:
        yield prefix, node

def extract_frozen_parameters(config:dict[str,Any])->list[FrozenParameter]:
    params=[]
    for _,node in _walk(config):
        if isinstance(node,dict) and {"key","value","category","source","rationale"}.issubset(node):
            params.append(FrozenParameter(str(node['key']), node['value'], str(node['category']), str(node['source']), str(node['rationale']), tuple(node.get('allowed_method_differences') or ())))
    return params

def validate_frozen_parameters(params:list[FrozenParameter])->None:
    seen=set()
    for p in params:
        if not p.key or p.value is None or not p.category or not p.source or not p.rationale: raise ParameterFreezeError(f"incomplete frozen parameter: {p.key}")
        if p.category not in CATEGORIES: raise ParameterFreezeError(f"unknown category for {p.key}: {p.category}")
        if p.key in seen: raise DuplicateFrozenParameterError(f"duplicate frozen parameter key: {p.key}")
        seen.add(p.key)

def unresolved_placeholders(config:dict[str,Any])->list[dict[str,str]]:
    out=[]
    for path,node in _walk(config):
        if _is_placeholder(node):
            low=path.lower()+" "+node.lower(); status="BLOCKED_UNRESOLVED_PARAMETER"
            if "bank_hash" in low: status="BLOCKED_SCENARIO_BANK_HASH"
            elif "scale" in low: status="BLOCKED_REWARD_SCALE_HASH"
            elif "reward" in low or "checkpoint_hash" in low: status="BLOCKED_REWARD_CHECKPOINT_HASH"
            elif "policy" in low: status="BLOCKED_POLICY_CONFIGURATION"
            out.append({"path":path,"value":node,"resolved_status":"unresolved_placeholder","blocked_status":status})
    return out

def validate_reward_weights(block:dict[str,Any])->None:
    comps=block.get('components',{})
    keys=set(comps)
    missing=set(CANONICAL_REWARD_COMPONENTS)-keys; unknown=keys-set(CANONICAL_REWARD_COMPONENTS)
    if missing: raise ParameterFreezeError(f"missing reward components: {sorted(missing)}")
    if unknown: raise ParameterFreezeError(f"unknown reward components: {sorted(unknown)}")
    weights=[]
    for k,v in comps.items():
        for req in ('weight','physical_unit','sign_convention','source','rationale'):
            if req not in v: raise ParameterFreezeError(f"reward component {k} missing {req}")
        w=float(v['weight'])
        if w < 0 or w != w or w in (float('inf'), float('-inf')): raise ParameterFreezeError(f"invalid reward weight {k}")
        weights.append(w)
    if not any(w>0 for w in weights): raise ParameterFreezeError('reward weights are all zero')

def expected_benchmark_rows(evaluation_protocol:dict[str,Any], seeds:list[int])->dict[str,int|str]:
    methods=evaluation_protocol['method_matrix']
    h=[m for m in methods['heuristics'] if m.get('enabled',True)]
    l=[m for m in methods['learned'] if m.get('enabled',True)]
    n=int(evaluation_protocol['test_scenario_count'])
    return {'formula':'heuristics * test_scenarios + learned_methods * training_seeds * test_scenarios','heuristic_rows':len(h)*n,'learned_rows':len(l)*len(seeds)*n,'total':len(h)*n+len(l)*len(seeds)*n}

def canonical_hash(payload:Any)->str:
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()

def freeze_hash(config:dict[str,Any])->str:
    clean=copy.deepcopy(config); clean.pop('freeze_hash',None)
    return canonical_hash(clean)

def git_metadata()->dict[str,Any]:
    def run(args):
        try: return subprocess.check_output(args,text=True).strip()
        except Exception: return None
    return {'code_commit':run(['git','rev-parse','HEAD']),'dirty_status':run(['git','status','--short']) or ''}

def validate_freeze_template(config:dict[str,Any], *, readiness:bool=False)->dict[str,Any]:
    params=extract_frozen_parameters(config); validate_frozen_parameters(params)
    if tuple(config['scientific_contract']['agents']['value']) != CANONICAL_AGENTS: raise ParameterFreezeError('canonical agents changed')
    if config['scientific_contract']['decision_events']['value'] != DECISION_EVENT_TO_AGENT: raise ParameterFreezeError('decision events changed')
    if tuple(config['scientific_contract']['station_baseline_actions']['value']) != STATION_BASELINE_ACTIONS: raise ParameterFreezeError('station actions changed')
    if config['paper_method_contract']['primary_rlaif_method']['value'] != 'mappo_rlaif_assignment': raise ParameterFreezeError('primary RLAIF method changed')
    if config['paper_method_contract']['reward_model_role']['value'] != 'score_selected_transition_only': raise ParameterFreezeError('reward model role changed')
    validate_reward_weights(config['reward_weights'])
    if config['scenario_protocol']['test_scenario_count']['value'] != 100: raise ParameterFreezeError('test count must be 100')
    if config['seed_protocol']['training_seeds']['value'] != [1,2,3]: raise ParameterFreezeError('training seeds must be [1, 2, 3]')
    if config['training_protocol']['mappo_optimization']['total_training_episodes']['value'] != 3000: raise ParameterFreezeError('MAPPO total episodes must be 3000')
    placeholders=unresolved_placeholders(config)
    blocked=sorted({p['blocked_status'] for p in placeholders})
    status='PARAMETER_FREEZE_READY' if not placeholders else 'PARAMETER_FREEZE_TEMPLATE_VALID'
    if readiness and placeholders: raise UnresolvedParameterFreezeError('unresolved placeholders block readiness')
    rows=expected_benchmark_rows(config['evaluation_protocol'], config['seed_protocol']['training_seeds']['value'])
    return {'status':status,'blocked_statuses':blocked,'unresolved_placeholders':placeholders,'parameter_count':len(params),'expected_benchmark_rows':rows,'freeze_hash':freeze_hash(config)}
