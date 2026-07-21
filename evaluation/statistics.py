"""Reproducible statistical aggregation for paper experiments."""
from __future__ import annotations
import math, random
from statistics import fmean, median, stdev
from typing import Iterable

def _success(rows): return [r for r in rows if r.get('status')=='success']

def bootstrap_ci(values: Iterable[float], *, seed:int=0, confidence:float=.95, samples:int=1000):
    vals=[float(v) for v in values if isinstance(v,(int,float)) and math.isfinite(float(v))]
    if not vals: return (None,None)
    rng=random.Random(seed); means=[]
    for _ in range(samples): means.append(fmean(rng.choice(vals) for _ in vals))
    means.sort(); alpha=(1-confidence)/2
    return (means[int(alpha*samples)], means[min(samples-1,int((1-alpha)*samples))])

def summarize_metric(rows, metric:str, *, seed:int=0):
    ok=_success(rows); vals=[float(r[metric]) for r in ok if isinstance(r.get(metric),(int,float)) and math.isfinite(float(r[metric]))]
    lo,hi=bootstrap_ci(vals,seed=seed)
    return {"metric":metric,"mean":fmean(vals) if vals else None,"std":stdev(vals) if len(vals)>1 else (0.0 if vals else None),"median":median(vals) if vals else None,"ci95_low":lo,"ci95_high":hi,"success_count":len(ok),"failure_count":sum(r.get('status')=='failed' for r in rows),"skip_count":sum(str(r.get('status','')).startswith('skipped') for r in rows)}

def paired_difference(a_rows,b_rows,metric:str,*,seed:int=0):
    a={r['scenario_id']:r for r in _success(a_rows)}; b={r['scenario_id']:r for r in _success(b_rows)}
    if set(a)!=set(b): raise ValueError('paired comparison requires matched scenario_id sets')
    diffs=[float(a[s][metric])-float(b[s][metric]) for s in sorted(a)]
    lo,hi=bootstrap_ci(diffs,seed=seed)
    sd=stdev(diffs) if len(diffs)>1 else 0.0
    return {"metric":metric,"paired_mean_difference":fmean(diffs) if diffs else None,"paired_ci95_low":lo,"paired_ci95_high":hi,"effect_size":(fmean(diffs)/sd if sd else 0.0),"paired_test":"sign_test" if len(diffs)>1 else "not_valid","n_pairs":len(diffs)}
