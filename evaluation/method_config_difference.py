"""Strict method resolved-config difference validation."""
from __future__ import annotations
from typing import Any

class UnexpectedMethodConfigDifferenceError(RuntimeError): pass

def _flat(x:Any,prefix=''):
    if isinstance(x,dict):
        out={}
        for k,v in x.items(): out.update(_flat(v, f'{prefix}.{k}' if prefix else str(k)))
        return out
    return {prefix:x}

def _match(path:str, patterns:list[str])->bool:
    return any(path==p or path.startswith(p+'.') or p.endswith('*') and path.startswith(p[:-1]) for p in patterns)

def _pair(contract:dict, a:str,b:str)->dict:
    pairs=contract.get('comparisons',{})
    return pairs.get(f'{a}__vs__{b}') or pairs.get(f'{b}__vs__{a}') or {}

def validate_method_config_differences(baseline:dict, comparison:dict, *, baseline_method:str, comparison_method:str, contract:dict)->dict:
    spec=_pair(contract, baseline_method, comparison_method)
    allowed=spec.get('allowed_differences',[]); expected=spec.get('expected_differences',[])
    fa,fb=_flat(baseline),_flat(comparison); keys=set(fa)|set(fb)
    diffs=sorted(k for k in keys if fa.get(k)!=fb.get(k))
    unexpected=[k for k in diffs if not _match(k,allowed)]
    missing=[]
    for p in expected:
        present=any(k==p or k.startswith(p+'.') for k in keys)
        if present and not any(_match(d,[p]) for d in diffs): missing.append(p)
    status='passed' if not unexpected and not missing else 'failed'
    report={'baseline_method':baseline_method,'comparison_method':comparison_method,'allowed_differences':allowed,'actual_differences':diffs,'unexpected_differences':unexpected,'missing_expected_differences':missing,'comparison_status':status}
    if status!='passed': raise UnexpectedMethodConfigDifferenceError(str(report))
    return report
