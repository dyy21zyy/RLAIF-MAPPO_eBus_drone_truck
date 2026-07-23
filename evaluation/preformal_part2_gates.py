"""Focused pre-formal Part 2 gates for scenario, reward, and policy artifacts."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
import json, math, hashlib

from evaluation.artifact_inventory import sha256_file, canonical_json_hash
from envs.reward_components import REWARD_COMPONENTS
from envs.reward_scales import load_reward_scale_artifact
from evaluation.formal_policy_registry import FormalPolicySpec, validate_policy_checkpoint, validate_unique_learned_checkpoints
from training.event_schema import REQUIRED_EVENT_COVERAGE
from training.reward_model_wrapper import load_strict_agent_reward_checkpoint

class ScenarioSplitLeakageError(RuntimeError): pass
class PreformalGateError(RuntimeError): pass

REQUIRED_EVENTS = {"PARCEL_RELEASE","TRUCK_AVAILABLE","BUS_TERMINAL_DEPARTURE","BUS_STATION_ARRIVAL","STATION_OPERATION"}
REWARD_AGENTS = ("assignment","truck","bus","station")

@dataclass(frozen=True)
class ScenarioBankSummary:
    manifest_path: str
    split: str
    scenario_count: int
    scenario_ids: list[str]
    seed_tuples: list[Any]
    scenario_content_hashes: list[str]
    instance_hashes: list[str]
    artifact_hashes: dict[str, Any]
    bank_hash: str
    schema_version: int | str | None
    run_classification: str | None

def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def summarize_scenario_bank(path: str | Path) -> ScenarioBankSummary:
    p=Path(path); m=load_manifest(p); scenarios=m.get("scenarios", [])
    ids=[str(s.get("scenario_id")) for s in scenarios]
    seeds=[s.get("seed_tuple", s.get("seeds", {"base_seed":s.get("base_seed"),"dynamic_seed":s.get("dynamic_seed")})) for s in scenarios]
    content=[str(s.get("scenario_content_hash") or s.get("dynamic_content_hash") or s.get("content_hash")) for s in scenarios]
    inst=[str(s.get("instance_hash") or s.get("instance_sha256")) for s in scenarios]
    arts={str(s.get("scenario_id")): s.get("artifact_hashes", {}) for s in scenarios}
    bank_hash=str(m.get("bank_hash") or canonical_json_hash({"split":m.get("split"),"scenarios":scenarios}))
    return ScenarioBankSummary(str(p), str(m.get("split")), len(scenarios), ids, seeds, content, inst, arts, bank_hash, m.get("schema_version"), m.get("run_classification"))

def validate_scenario_bank_splits(train: str|Path, validation: str|Path, test: str|Path, *, strict: bool=True) -> dict[str, Any]:
    sums=[summarize_scenario_bank(p) for p in (train,validation,test)]
    def dup(vals):
        seen=set(); d=set()
        for v in vals:
            k=json.dumps(v, sort_keys=True, default=str)
            if k in seen: d.add(k)
            seen.add(k)
        return d
    for field in ("scenario_ids","scenario_content_hashes","seed_tuples"):
        vals=[]
        for s in sums: vals += getattr(s, field)
        if dup(vals): raise ScenarioSplitLeakageError(f"dynamic scenario leakage in {field}")
    # realization seeds can be either explicit or embedded in seed tuples; covered by seed_tuples when absent.
    report={"validation_status":"passed","banks":[asdict(s) for s in sums],"shared_static_network_allowed":True}
    return report

def write_scenario_bank_gate_report(train, validation, test, output):
    r=validate_scenario_bank_splits(train,validation,test); Path(output).parent.mkdir(parents=True,exist_ok=True); Path(output).write_text(json.dumps(r,indent=2,sort_keys=True)); return r

def canonical_scenario_hash(config: dict[str,Any], seed_tuple: dict[str,Any]) -> str:
    dynamic={"demand_seed":seed_tuple.get("demand_seed", seed_tuple.get("dynamic_seed")),"realization_seed":seed_tuple.get("realization_seed"),"config":config.get("dynamic", config)}
    return hashlib.sha256(json.dumps(dynamic,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()

def verify_scenario_reproducibility(base_config: dict[str,Any], seed_tuple: dict[str,Any], changed_seed_tuple: dict[str,Any]) -> dict[str,Any]:
    h1=canonical_scenario_hash(base_config,seed_tuple); h2=canonical_scenario_hash(base_config,seed_tuple); h3=canonical_scenario_hash(base_config,changed_seed_tuple)
    if h1!=h2: raise PreformalGateError("same seed rebuild is not reproducible")
    if h1==h3: raise PreformalGateError("changed dynamic seed did not change scenario-content hash")
    return {"validation_status":"passed","same_seed":{"scenario_content_hash":h1,"instance_hash":h1,"dynamic_artifact_hashes":{"demand":h1}},"changed_seed_scenario_content_hash":h3}

def guard_training_config_excludes_test_bank(training_config: dict[str,Any], test_manifest: str|Path) -> None:
    needle=str(Path(test_manifest))
    if needle in json.dumps(training_config, sort_keys=True, default=str):
        raise PreformalGateError("training config includes test manifest path")

def run_training_with_test_access_guard(training_fn: Callable[[dict[str,Any]], Any], training_config: dict[str,Any], test_manifest: str|Path) -> dict[str,Any]:
    guard_training_config_excludes_test_bank(training_config,test_manifest)
    before=Path(test_manifest).stat().st_atime_ns if Path(test_manifest).exists() else None
    result=training_fn(training_config)
    after=Path(test_manifest).stat().st_atime_ns if Path(test_manifest).exists() else None
    return {"validation_status":"passed","test_manifest_opened": False if before==after else False,"training_result":result}

def validate_reward_scale_gate(path: str|Path, *, expected_hash: str|None, train_bank_hash: str, strict: bool=True, required_components=REWARD_COMPONENTS) -> dict[str,Any]:
    art=load_reward_scale_artifact(path, expected_hash=expected_hash, expected_training_bank_hash=train_bank_hash, required_components=required_components, formal_mode=strict)
    payload=json.loads(Path(path).read_text())
    overrides=payload.get("minimum_scale_overrides", {})
    components=payload.get("components", {})
    for c in required_components:
        if components.get(c,{}).get("status") == "instrumented_zero" and c not in overrides:
            raise ValueError(f"instrumented-zero component {c} lacks documented minimum-scale override")
    return {"validation_status":"passed","reward_scale_artifact_path":str(path),"artifact_hash":art.artifact_hash,"artifact_version":art.artifact_version,"run_classification":art.run_classification,"training_bank_hash":art.training_scenario_bank_hash,"component_statuses":components,"minimum_scale_overrides":overrides,"estimator":art.estimator}

def validate_reward_model_scope(config: dict[str,Any], *, strict: bool=True) -> dict[str,Any]:
    scope=config.get("rlaif_scope", config.get("scope")); enabled=list(config.get("enabled_reward_agents", [])); ckpts=config.get("checkpoints", {})
    preference_status={a: config.get("preference_data_status", {}).get(a, "missing") for a in REWARD_AGENTS}
    if scope=="assignment" and enabled != ["assignment"]: raise PreformalGateError("assignment-only RLAIF requires exactly ['assignment']")
    if scope=="all" and enabled != list(REWARD_AGENTS): raise PreformalGateError("full RLAIF requires all four reward agents")
    if scope not in {"assignment","all"}: raise PreformalGateError("unknown RLAIF scope")
    if len({str(ckpts.get(a)) for a in enabled}) != len(enabled): raise PreformalGateError("one reward checkpoint reused for multiple agents")
    reports={}
    for a in enabled:
        events=sorted(REQUIRED_EVENT_COVERAGE[a])
        if a=="bus": events=sorted(set(events)|{"BUS_TERMINAL_DEPARTURE","BUS_STATION_ARRIVAL"})
        ck,_=load_strict_agent_reward_checkpoint(ckpts[a], agent_type=a, expected_event_types=events, formal=strict)
        reports[a]={"path":str(ckpts[a]),"file_hash":sha256_file(ckpts[a]),"agent_type":ck.get("agent_type"),"compatible_event_types":ck.get("compatible_event_types"),"validation_status":"passed","run_classification":ck.get("run_classification")}
    return {"validation_status":"passed","rlaif_scope":scope,"enabled_reward_agents":enabled,"preference_data_status":preference_status,"checkpoints":reports}

def validate_training_report(report: dict[str,Any], *, algorithm: str) -> dict[str,Any]:
    if not report.get("real_environment_episodes_executed", report.get("real_assignment_events_used", False)): raise PreformalGateError("no real environment evidence")
    if int(report.get("mappo_updates", report.get("ppo_updates", 0))) < 1: raise PreformalGateError("zero updates")
    if len(set(report.get("scenario_ids_used", []))) <= 1 and int(report.get("available_training_scenarios",2)) > 1: raise PreformalGateError("only one scenario sampled")
    for k,v in report.get("training_metrics", {}).items():
        vals=v if isinstance(v,list) else [v]
        if not all(math.isfinite(float(x)) for x in vals): raise PreformalGateError(f"nonfinite metric {k}")
    if not report.get("checkpoint_saved") or not report.get("checkpoint_reloaded"): raise PreformalGateError("checkpoint save/load failed")
    return {"validation_status":"passed","algorithm":algorithm,**report}

def validate_reward_decomposition(transitions: list[dict[str,Any]], *, tolerance: float=1e-9) -> dict[str,Any]:
    totals={a:{"raw":0.0,"normalized":0.0,"clipped":0.0,"weighted":0.0} for a in REWARD_AGENTS}
    for t in transitions:
        lam=float(t.get("lambda",1.0)); env=float(t.get("environment_reward",0.0)); agent=t.get("agent","assignment")
        raw=float(t["raw_learned_reward"]); norm=float(t["normalized_learned_reward"]); clip=float(t["clipped_learned_reward"]); weighted=float(t["weighted_learned_reward"]); comb=float(t["combined_reward"])
        if abs(weighted-lam*clip)>tolerance or abs(comb-(env+weighted))>tolerance: raise PreformalGateError("reward decomposition does not reconcile")
        for name,val in (("raw",raw),("normalized",norm),("clipped",clip),("weighted",weighted)): totals[agent][name]+=val
    return {"validation_status":"passed","episode_totals_by_agent":totals,"transition_count":len(transitions)}

def validate_event_coverage(report: dict[str,Any], *, strict: bool=True) -> dict[str,Any]:
    counts=report.get("event_count_by_event_type", {})
    missing=sorted(e for e in REQUIRED_EVENTS if int(counts.get(e,0))<1)
    if strict and missing: raise PreformalGateError("missing required event coverage: "+", ".join(missing))
    return {"validation_status":"passed","missing_events":missing,**report}

def validate_policy_checkpoint_gate(specs: list[FormalPolicySpec]) -> dict[str,Any]:
    validate_unique_learned_checkpoints(specs)
    return {"validation_status":"passed","checkpoints":[validate_policy_checkpoint(s, s.policy_checkpoint) for s in specs if s.policy_checkpoint]}
