from __future__ import annotations
from typing import Any

class UnexpectedSensitivityConfigDifferenceError(RuntimeError): pass
IGNORED_SUFFIXES=("path","paths","hash","hashes","timestamp","time","id","ids","output_root","output")
IGNORED_EXACT={"created_at","updated_at","resolved_at","run_id","job_id"}

def _flatten(obj: Any, prefix="") -> dict[str, Any]:
    if isinstance(obj, dict):
        out={}
        for k,v in obj.items(): out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: obj}

def _ignored(path: str) -> bool:
    leaf=path.split('.')[-1]
    return leaf in IGNORED_EXACT or any(leaf.endswith(s) for s in IGNORED_SUFFIXES) or 'artifact' in path or 'scenario' in path

def validate_sensitivity_config_difference(baseline: dict[str,Any], candidate: dict[str,Any], declared_path: str) -> None:
    b,c=_flatten(baseline),_flatten(candidate)
    bad=[]
    for p in sorted(set(b)|set(c)):
        if b.get(p)!=c.get(p) and p != declared_path and not _ignored(p): bad.append((p,b.get(p),c.get(p)))
    if bad: raise UnexpectedSensitivityConfigDifferenceError(f"unexpected sensitivity config differences: {bad}")

def validate_ablation_override_applied(baseline: dict[str,Any], candidate: dict[str,Any], declared_paths: list[str]|tuple[str,...]|set[str]) -> None:
    b,c=_flatten(baseline),_flatten(candidate); paths=list(declared_paths)
    if not paths or not any(b.get(p)!=c.get(p) for p in paths): raise UnexpectedSensitivityConfigDifferenceError("ablation override did not change resolved config")
