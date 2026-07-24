from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Any, Sequence
import hashlib, json
import torch
from envs.state_builder import CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES, BUS_EVENT_STATE_FEATURE_NAMES
from rlaif.feature_alignment import align_named_features
from training.event_schema import decision_event_id, REQUIRED_EVENT_COVERAGE
from rlaif.preference_schema_v3 import PreferenceRecord, parse_preference_record, resolve_original_binary_target, validate_label_source

@dataclass(frozen=True)
class RewardPairExample:
    preference_id: str; scenario_id: str; episode_id: str; state_id: str; agent_type: str; event_type_id: int
    state_features: torch.Tensor; candidate_a_features: torch.Tensor; candidate_b_features: torch.Tensor
    target_a_preferred: torch.Tensor; label_source: str

@dataclass
class DatasetIntegrityReport:
    total_records:int=0; usable_binary_records:int=0; excluded_outcomes:Counter=field(default_factory=Counter)
    duplicate_count:int=0; contradiction_count:int=0; duplicate_preference_ids:int=0; self_comparisons:int=0
    feature_schema_mismatches:int=0; nonfinite_fields:int=0; missing_fields:int=0
    counts_by_agent:Counter=field(default_factory=Counter); counts_by_event:Counter=field(default_factory=Counter); counts_by_label_source:Counter=field(default_factory=Counter); counts_by_outcome:Counter=field(default_factory=Counter)
    def to_dict(self):
        return {"total_records":self.total_records,"usable_binary_records":self.usable_binary_records,"ties":self.excluded_outcomes.get('tie',0),"abstentions":self.excluded_outcomes.get('abstain',0),"invalid":self.excluded_outcomes.get('invalid',0),"unresolved":self.excluded_outcomes.get('unresolved',0),"duplicate_preference_ids":self.duplicate_preference_ids,"exact_duplicate_pairs":self.duplicate_count,"contradictory_pairs":self.contradiction_count,"self_comparisons":self.self_comparisons,"feature_schema_mismatches":self.feature_schema_mismatches,"nonfinite_fields":self.nonfinite_fields,"missing_fields":self.missing_fields,"counts_by_agent":dict(self.counts_by_agent),"counts_by_event":dict(self.counts_by_event),"counts_by_label_source":dict(self.counts_by_label_source),"counts_by_outcome":dict(self.counts_by_outcome)}

class RewardPairDataset(torch.utils.data.Dataset):
    def __init__(self, examples: Sequence[RewardPairExample], *, agent_type:str, state_feature_names:tuple[str,...], candidate_feature_names:tuple[str,...], report:DatasetIntegrityReport|None=None):
        self.examples=list(examples); self.agent_type=agent_type; self.state_feature_names=state_feature_names; self.candidate_feature_names=candidate_feature_names; self.state_dim=len(state_feature_names); self.candidate_dim=len(candidate_feature_names); self.report=report or DatasetIntegrityReport()
    def __len__(self): return len(self.examples)
    def __getitem__(self,i): return self.examples[i]

def _pair_key(r: PreferenceRecord): return (r.scenario_id,r.episode_id,r.state_id,r.agent_type,r.event_type,frozenset({r.original_candidate_a_id,r.original_candidate_b_id}))
def _cand_key(r: PreferenceRecord, cid: str): return (r.scenario_id,r.episode_id,r.state_id,r.agent_type,r.event_type,cid)


def _canonicalize_record_features(r: PreferenceRecord) -> tuple[tuple[str, ...], tuple[float, ...], tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    state_names = r.state_feature_names
    state_features = r.state_features
    cand_names = r.candidate_a_feature_names
    cand_a = r.candidate_a_features
    cand_b = r.candidate_b_features
    if r.agent_type == 'bus':
        source_state_names = BUS_EVENT_STATE_FEATURE_NAMES.get(r.event_type)
        if source_state_names is None:
            raise ValueError(f'unsupported bus event type {r.event_type}')
        if state_names != source_state_names and state_names != CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES:
            raise ValueError('state feature order mismatch')
        state_features = align_named_features(
            state_names, state_features, CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES
        )
        state_names = CANONICAL_BUS_REWARD_STATE_FEATURE_NAMES
    return state_names, state_features, cand_names, cand_a, cand_b

def build_reward_pair_dataset(rows: Sequence[dict[str,Any]|PreferenceRecord], *, agent_type:str, formal_mode:bool=True, expected_state_feature_names:tuple[str,...]|None=None, expected_candidate_feature_names:tuple[str,...]|None=None, require_bus_event_coverage:bool=False) -> RewardPairDataset:
    report=DatasetIntegrityReport(total_records=len(rows)); records=[]; ids=set()
    for row in rows:
        try:
            r = row if isinstance(row,PreferenceRecord) else parse_preference_record(row, formal_mode=formal_mode)
            validate_label_source(r.label_source, formal_mode=formal_mode)
            if r.preference_id in ids: report.duplicate_preference_ids += 1
            ids.add(r.preference_id)
            report.counts_by_agent[r.agent_type]+=1; report.counts_by_event[r.event_type]+=1; report.counts_by_label_source[r.label_source]+=1; report.counts_by_outcome[r.original_outcome]+=1
            if r.agent_type != agent_type: continue
            if r.original_candidate_a_id == r.original_candidate_b_id: report.self_comparisons += 1; raise ValueError('self-comparison')
            state_names, state_features, cand_names, cand_a, cand_b = _canonicalize_record_features(r)
            if expected_state_feature_names and state_names != expected_state_feature_names: report.feature_schema_mismatches += 1; raise ValueError('state feature order mismatch')
            if expected_candidate_feature_names and cand_names != expected_candidate_feature_names: report.feature_schema_mismatches += 1; raise ValueError('candidate feature order mismatch')
            records.append((r, state_names, state_features, cand_names, cand_a, cand_b))
        except KeyError as e: report.missing_fields+=1; raise ValueError(f"missing field {e}") from e
        except ValueError as e:
            msg=str(e)
            if 'non-finite' in msg: report.nonfinite_fields += 1
            if 'feature' in msg and 'mismatch' in msg: report.feature_schema_mismatches += 1
            raise
    if require_bus_event_coverage and agent_type=='bus':
        missing=REQUIRED_EVENT_COVERAGE['bus'] - {r.event_type for r, *_ in records}
        if missing: raise ValueError(f"formal bus validation missing required events: {sorted(missing)}")
    if not records: return RewardPairDataset([],agent_type=agent_type,state_feature_names=(),candidate_feature_names=(),report=report)
    state_names=records[0][1]; cand_names=records[0][3]
    cand_features={}; grouped=defaultdict(list)
    for r, rec_state_names, rec_state_features, rec_cand_names, rec_cand_a, rec_cand_b in records:
        if rec_state_names!=state_names or rec_cand_names!=cand_names or r.candidate_b_feature_names!=cand_names:
            report.feature_schema_mismatches += 1; raise ValueError('feature schemas conflict')
        for cid, feats in ((r.original_candidate_a_id,rec_cand_a),(r.original_candidate_b_id,rec_cand_b)):
            k=_cand_key(r,cid)
            if k in cand_features and cand_features[k] != feats: raise ValueError('same candidate ID has inconsistent features in the same state')
            cand_features[k]=feats
        target=resolve_original_binary_target(r)
        if target is None: report.excluded_outcomes[r.original_outcome]+=1; continue
        grouped[_pair_key(r)].append((r,target,rec_state_features,rec_cand_a,rec_cand_b))
    examples=[]
    for vals in grouped.values():
        targets={t for _,t,_,_,_ in vals}
        if len(targets)>1: report.contradiction_count += len(vals); continue
        if len(vals)>1: report.duplicate_count += len(vals)-1
        r,t,state_features,cand_a,cand_b=vals[0]
        examples.append(RewardPairExample(r.preference_id,r.scenario_id,r.episode_id,r.state_id,r.agent_type,decision_event_id(r.event_type),torch.tensor(state_features,dtype=torch.float32),torch.tensor(cand_a,dtype=torch.float32),torch.tensor(cand_b,dtype=torch.float32),torch.tensor(float(t),dtype=torch.float32),r.label_source))
    report.usable_binary_records=len(examples)
    return RewardPairDataset(examples,agent_type=agent_type,state_feature_names=state_names,candidate_feature_names=cand_names,report=report)

def dataset_hash(ds: RewardPairDataset)->str:
    payload=[(e.preference_id,e.scenario_id,e.episode_id,e.state_id,e.agent_type,e.event_type_id,e.state_features.tolist(),e.candidate_a_features.tolist(),e.candidate_b_features.tolist(),float(e.target_a_preferred)) for e in ds.examples]
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
