from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal
import math
from training.event_schema import (OBSERVATION_SCHEMA_VERSION, CANDIDATE_SCHEMA_VERSION, EVENT_SCHEMA_VERSION, DECISION_EVENT_SPECS, normalize_decision_event_type, validate_agent_event)
PREFERENCE_SCHEMA_VERSION = 3
AgentType = Literal["assignment","truck","bus","station"]
PreferenceOutcome = Literal["candidate_a","candidate_b","tie","abstain","invalid","unresolved"]
SUPPORTED_AGENTS = frozenset({"assignment","truck","bus","station"})
SUPPORTED_OUTCOMES = frozenset({"candidate_a","candidate_b","tie","abstain","invalid","unresolved"})
FORMAL_LABEL_SOURCES = frozenset({"external_evaluator_api","validated_replay"})
SMOKE_LABEL_SOURCES = frozenset({"synthetic_smoke"})
AGENT_EVENT_TYPES = {a: frozenset(k for k,v in DECISION_EVENT_SPECS.items() if v.agent_type==a) for a in SUPPORTED_AGENTS}
@dataclass(frozen=True)
class PreferenceRecord:
    preference_id: str; scenario_id: str; episode_id: str; state_id: str
    agent_type: AgentType; event_type: str
    state_feature_names: tuple[str,...]; state_features: tuple[float,...]
    candidate_a_feature_names: tuple[str,...]; candidate_a_features: tuple[float,...]
    candidate_b_feature_names: tuple[str,...]; candidate_b_features: tuple[float,...]
    original_candidate_a_id: str; original_candidate_b_id: str
    displayed_first_candidate_id: str; displayed_second_candidate_id: str
    original_outcome: PreferenceOutcome
    label_source: str; evaluator_model: str|None; evaluator_prompt_version: str|None
    observation_schema_version: int; candidate_schema_version: int; event_schema_version: int
    created_at: str

def _tuple_str(x: Any) -> tuple[str,...]: return tuple(str(v) for v in x)
def _tuple_float(x: Any) -> tuple[float,...]: return tuple(float(v) for v in x)
def validate_label_source(source: str, *, formal_mode: bool) -> None:
    if formal_mode:
        if source not in FORMAL_LABEL_SOURCES: raise ValueError(f"label_source {source!r} is not allowed for formal reward data")
    elif source not in FORMAL_LABEL_SOURCES | SMOKE_LABEL_SOURCES:
        raise ValueError(f"unknown label_source {source!r}")
def parse_preference_record(row: dict[str,Any], *, formal_mode: bool|None=None) -> PreferenceRecord:
    missing=[k for k in PreferenceRecord.__dataclass_fields__ if k not in row]
    if missing: raise ValueError(f"preference record missing fields: {missing}")
    rec=PreferenceRecord(
        preference_id=str(row['preference_id']), scenario_id=str(row['scenario_id']), episode_id=str(row['episode_id']), state_id=str(row['state_id']),
        agent_type=str(row['agent_type']), event_type=normalize_decision_event_type(row['event_type']),
        state_feature_names=_tuple_str(row['state_feature_names']), state_features=_tuple_float(row['state_features']),
        candidate_a_feature_names=_tuple_str(row['candidate_a_feature_names']), candidate_a_features=_tuple_float(row['candidate_a_features']),
        candidate_b_feature_names=_tuple_str(row['candidate_b_feature_names']), candidate_b_features=_tuple_float(row['candidate_b_features']),
        original_candidate_a_id=str(row['original_candidate_a_id']), original_candidate_b_id=str(row['original_candidate_b_id']),
        displayed_first_candidate_id=str(row['displayed_first_candidate_id']), displayed_second_candidate_id=str(row['displayed_second_candidate_id']),
        original_outcome=str(row['original_outcome']), label_source=str(row['label_source']), evaluator_model=row.get('evaluator_model'), evaluator_prompt_version=row.get('evaluator_prompt_version'),
        observation_schema_version=int(row['observation_schema_version']), candidate_schema_version=int(row['candidate_schema_version']), event_schema_version=int(row['event_schema_version']), created_at=str(row['created_at']))
    validate_preference_record(rec)
    if formal_mode is not None: validate_label_source(rec.label_source, formal_mode=formal_mode)
    return rec
def validate_preference_record(record: PreferenceRecord) -> None:
    for name in ('preference_id','scenario_id','episode_id','state_id','original_candidate_a_id','original_candidate_b_id','displayed_first_candidate_id','displayed_second_candidate_id','created_at'):
        if not getattr(record,name): raise ValueError(f"{name} must be nonempty")
    if record.agent_type not in SUPPORTED_AGENTS: raise ValueError(f"unsupported agent_type {record.agent_type}")
    validate_agent_event(record.agent_type, record.event_type)
    if record.original_outcome not in SUPPORTED_OUTCOMES: raise ValueError(f"unsupported original_outcome {record.original_outcome}")
    if record.original_candidate_a_id == record.original_candidate_b_id: raise ValueError("candidate A and B must differ")
    if {record.displayed_first_candidate_id,record.displayed_second_candidate_id}!={record.original_candidate_a_id,record.original_candidate_b_id}: raise ValueError("displayed candidate IDs must match original A/B IDs")
    if record.observation_schema_version != OBSERVATION_SCHEMA_VERSION or record.candidate_schema_version != CANDIDATE_SCHEMA_VERSION or record.event_schema_version != EVENT_SCHEMA_VERSION: raise ValueError("unsupported schema version")
    for names, vals, label in ((record.state_feature_names,record.state_features,'state'),(record.candidate_a_feature_names,record.candidate_a_features,'candidate_a'),(record.candidate_b_feature_names,record.candidate_b_features,'candidate_b')):
        if len(names)!=len(vals): raise ValueError(f"{label} feature names and values differ")
        if not names: raise ValueError(f"{label} feature names are empty")
        if any(not math.isfinite(v) for v in vals): raise ValueError(f"{label} features contain non-finite values")
    if record.candidate_a_feature_names != record.candidate_b_feature_names: raise ValueError("candidate feature name order mismatch")
def resolve_original_binary_target(record: PreferenceRecord) -> int|None:
    return 1 if record.original_outcome=='candidate_a' else 0 if record.original_outcome=='candidate_b' else None
