"""Canonical formal method registry and checkpoint-lineage validation."""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal
import hashlib, json

MethodType = Literal['truck_direct_heuristic','integrated_rule_based','assignment_ppo','mappo_env','mappo_rlaif_assignment','mappo_rlaif_all']
REWARD_AGENTS=('assignment','truck','bus','station')
LEARNED_METHODS=('assignment_ppo','mappo_env','mappo_rlaif_assignment','mappo_rlaif_all')
EXPECTED_ALGORITHMS={
 'assignment_ppo':'assignment_ppo',
 'mappo_env':'four_agent_asynchronous_mappo_env',
 'mappo_rlaif_assignment':'four_agent_asynchronous_mappo_rlaif_assignment',
 'mappo_rlaif_all':'four_agent_asynchronous_mappo_rlaif_all',
}
EXPECTED_SCOPE={'truck_direct_heuristic':'none','integrated_rule_based':'none','assignment_ppo':'none','mappo_env':'none','mappo_rlaif_assignment':'assignment','mappo_rlaif_all':'all'}

@dataclass(frozen=True)
class FormalPolicySpec:
    method_id: MethodType
    display_name: str
    policy_type: str
    policy_checkpoint: str | None = None
    expected_algorithm: str | None = None
    expected_rlaif_scope: str = 'none'
    enabled_reward_agents: tuple[str,...] = ()
    reward_checkpoints: dict[str,str] | None = None
    training_seed: int | None = None
    formal_mode: bool = True

    def with_checkpoint(self, checkpoint: str, *, training_seed:int|None=None, reward_checkpoints:dict[str,str]|None=None):
        return replace(self, policy_checkpoint=checkpoint, training_seed=training_seed if training_seed is not None else self.training_seed, reward_checkpoints=reward_checkpoints if reward_checkpoints is not None else self.reward_checkpoints)

FORMAL_METHOD_REGISTRY: dict[str, FormalPolicySpec] = {
 'truck_direct_heuristic': FormalPolicySpec('truck_direct_heuristic','Truck-direct heuristic','heuristic'),
 'integrated_rule_based': FormalPolicySpec('integrated_rule_based','Integrated rule-based heuristic','heuristic'),
 'assignment_ppo': FormalPolicySpec('assignment_ppo','Assignment PPO','assignment_ppo',expected_algorithm='assignment_ppo'),
 'mappo_env': FormalPolicySpec('mappo_env','Four-agent environment MAPPO','mappo',expected_algorithm='four_agent_asynchronous_mappo_env'),
 'mappo_rlaif_assignment': FormalPolicySpec('mappo_rlaif_assignment','Assignment-only RLAIF-MAPPO','mappo',expected_algorithm='four_agent_asynchronous_mappo_rlaif_assignment',expected_rlaif_scope='assignment',enabled_reward_agents=('assignment',)),
 'mappo_rlaif_all': FormalPolicySpec('mappo_rlaif_all','Full four-agent RLAIF-MAPPO','mappo',expected_algorithm='four_agent_asynchronous_mappo_rlaif_all',expected_rlaif_scope='all',enabled_reward_agents=REWARD_AGENTS),
}

class PolicyRegistryError(ValueError): pass
class PolicyCheckpointValidationError(RuntimeError): pass

def get_formal_policy_spec(method_id: str) -> FormalPolicySpec:
    try: return FORMAL_METHOD_REGISTRY[method_id]
    except KeyError as exc: raise PolicyRegistryError(f'unknown formal method: {method_id}') from exc

def _sha(path: str|Path) -> str:
    h=hashlib.sha256();
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def load_checkpoint_metadata(path: str|Path) -> dict[str,Any]:
    p=Path(path)
    if not p.is_file(): raise PolicyCheckpointValidationError(f'policy checkpoint missing: {p}')
    if 'smoke' in p.parts and 'formal' not in p.parts: raise PolicyCheckpointValidationError(f'smoke policy checkpoint rejected in formal mode: {p}')
    if p.suffix.lower()=='.json':
        data=json.loads(p.read_text())
    else:
        import torch
        data=torch.load(p,map_location='cpu',weights_only=False)
    if not isinstance(data,dict): raise PolicyCheckpointValidationError('checkpoint is not a dictionary')
    meta=data.get('metadata') if isinstance(data.get('metadata'),dict) else data
    meta=dict(meta); meta['checkpoint_hash']=_sha(p); meta['checkpoint_path']=str(p)
    return meta

def validate_policy_checkpoint(spec: FormalPolicySpec, metadata_or_path: dict[str,Any] | str | Path) -> dict[str,Any]:
    if spec.method_id not in LEARNED_METHODS: return {}
    meta = load_checkpoint_metadata(metadata_or_path) if not isinstance(metadata_or_path,dict) else dict(metadata_or_path)
    if spec.formal_mode and str(meta.get('run_classification','formal')) in {'smoke','diagnostic'}: raise PolicyCheckpointValidationError('smoke/diagnostic checkpoint rejected formally')
    if spec.expected_algorithm and meta.get('algorithm') != spec.expected_algorithm: raise PolicyCheckpointValidationError(f"checkpoint algorithm {meta.get('algorithm')} != {spec.expected_algorithm}")
    scope=meta.get('rlaif_scope', meta.get('RLAIF scope', 'none'))
    if scope != spec.expected_rlaif_scope: raise PolicyCheckpointValidationError(f'checkpoint RLAIF scope {scope} != {spec.expected_rlaif_scope}')
    agents=tuple(meta.get('enabled_reward_agents') or ())
    if tuple(agents) != tuple(spec.enabled_reward_agents): raise PolicyCheckpointValidationError(f'checkpoint reward agents {agents} != {spec.enabled_reward_agents}')
    if spec.training_seed is not None and int(meta.get('training_seed',-1)) != int(spec.training_seed): raise PolicyCheckpointValidationError('checkpoint training seed mismatch')
    for k in ('training_scenario_bank_hash','resolved_training_config_hash','code_commit'):
        if spec.formal_mode and not meta.get(k): raise PolicyCheckpointValidationError(f'missing checkpoint lineage field: {k}')
    if meta.get('checkpoint_schema_version') is None: raise PolicyCheckpointValidationError('missing checkpoint schema version')
    if spec.formal_mode and meta.get('validation_status') in {'smoke_only','diagnostic_only'}: raise PolicyCheckpointValidationError('non-formal validation status rejected formally')
    r_hashes=meta.get('reward_checkpoint_hashes') or {}
    for a in spec.enabled_reward_agents:
        if not r_hashes.get(a): raise PolicyCheckpointValidationError(f'missing reward checkpoint hash for {a}')
    return meta

def validate_unique_learned_checkpoints(specs: list[FormalPolicySpec]) -> None:
    seen: dict[str,str] = {}
    for s in specs:
        if s.method_id not in LEARNED_METHODS or not s.policy_checkpoint: continue
        key=str(Path(s.policy_checkpoint))
        if key in seen and seen[key] != s.method_id: raise PolicyCheckpointValidationError(f'one checkpoint assigned to two learned methods: {key}')
        seen[key]=s.method_id
