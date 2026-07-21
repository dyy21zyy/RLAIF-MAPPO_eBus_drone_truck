"""Paired evaluation helpers: all methods must share exact scenario IDs."""
from __future__ import annotations
from collections import defaultdict

def group_by_method(rows):
    grouped=defaultdict(list)
    for r in rows: grouped[r.get('method_name')].append(r)
    return dict(grouped)

def validate_paired_scenarios(rows):
    grouped=group_by_method(rows)
    ids={m:{r.get('scenario_id') for r in rs if r.get('status')=='success'} for m,rs in grouped.items()}
    nonempty={m:s for m,s in ids.items() if s}
    if not nonempty: return True
    first=next(iter(nonempty.values()))
    bad={m:sorted(s^first) for m,s in nonempty.items() if s!=first}
    if bad: raise ValueError(f"paired evaluation requires identical scenario_id sets: {bad}")
    return True
