"""Paired evaluation helpers requiring exact scenario artifact identity."""
from __future__ import annotations
from collections import defaultdict

class PairedScenarioMismatchError(RuntimeError): pass

def group_by_method(rows):
    grouped=defaultdict(list)
    for r in rows: grouped[r.get('method_id', r.get('method_name'))].append(r)
    return dict(grouped)

def _sig(row):
    return (row.get('scenario_id'), row.get('instance_hash'), row.get('scenario_manifest_hash'), tuple(sorted((row.get('artifact_hashes') or row.get('exogenous_artifact_hashes') or {}).items())))

def assert_pairable(a:dict,b:dict)->bool:
    if a.get('scenario_id') != b.get('scenario_id'): raise PairedScenarioMismatchError('different scenario IDs cannot be paired')
    for k in ('instance_hash','scenario_manifest_hash'):
        if a.get(k) != b.get(k): raise PairedScenarioMismatchError(f'scenario {a.get("scenario_id")} {k} mismatch')
    ah=a.get('artifact_hashes') or a.get('exogenous_artifact_hashes') or {}
    bh=b.get('artifact_hashes') or b.get('exogenous_artifact_hashes') or {}
    if ah != bh: raise PairedScenarioMismatchError('exogenous artifact hashes mismatch')
    return True

def validate_paired_scenarios(rows):
    grouped=group_by_method(rows); maps={m:{r.get('scenario_id'):r for r in rs if r.get('status')=='success'} for m,rs in grouped.items()}
    nonempty={m:s for m,s in maps.items() if s}
    if not nonempty: return True
    first_m, first = next(iter(nonempty.items()))
    for m,mp in nonempty.items():
        if set(mp)!=set(first): raise PairedScenarioMismatchError(f'paired evaluation requires identical scenario_id sets: {m}')
        for sid,row in mp.items(): assert_pairable(first[sid], row)
    return True
