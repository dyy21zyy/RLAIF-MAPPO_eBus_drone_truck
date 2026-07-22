from __future__ import annotations
from statistics import fmean, median, stdev
from math import sqrt
from typing import Any

def _metric(row, name):
    v=row.get(name)
    if v is None and isinstance(row.get('formal_metrics'),dict):
        v=row['formal_metrics'].get(name)
        if isinstance(v,dict): v=v.get('value')
    return None if v is None else float(v)
def stats(values):
    vals=[float(v) for v in values if v is not None]
    if not vals: return {'sample_count':0,'status':'insufficient_samples'}
    sd=stdev(vals) if len(vals)>1 else 0.0; se=sd/sqrt(len(vals)) if vals else None; ci=1.96*se if se is not None and len(vals)>1 else None
    return {'sample_count':len(vals),'mean':fmean(vals),'standard_deviation':sd,'median':median(vals),'minimum':min(vals),'maximum':max(vals),'standard_error':se,'ci95_low':None if ci is None else fmean(vals)-ci,'ci95_high':None if ci is None else fmean(vals)+ci,'status':'ok' if len(vals)>1 else 'insufficient_samples'}
def compatibility_key(row):
    return (row.get('experiment_kind'),row.get('protocol'),row.get('run_classification'),row.get('method_id') or row.get('variant_id'),row.get('policy_rlaif_scope'),row.get('scenario_family_design') or row.get('test_bank_hash') or row.get('scenario_bank_hash'),row.get('reward_scale_hash'),row.get('metric_definition_version','formal_result_v2'))
def aggregate_compatible(rows: list[dict[str,Any]], metrics=('environment_reward',)):
    groups={}
    for r in rows:
        if r.get('status') not in {'evaluation_success','success'}: continue
        groups.setdefault(compatibility_key(r),[]).append(r)
    return [{'compatibility_key':list(k),'metric':m,**stats([_metric(r,m) for r in rs])} for k,rs in groups.items() for m in metrics]
def paired_differences(rows, *, baseline_selector, comparison_selector, metric='environment_reward'):
    base={ (r.get('scenario_family_id') or r.get('scenario_id'), r.get('master_seed')): r for r in rows if baseline_selector(r) and r.get('status') in {'evaluation_success','success'} }
    out=[]
    for r in rows:
        if not comparison_selector(r) or r.get('status') not in {'evaluation_success','success'}: continue
        key=(r.get('scenario_family_id') or r.get('scenario_id'), r.get('master_seed'))
        if key in base:
            bv,cv=_metric(base[key],metric),_metric(r,metric)
            if bv is not None and cv is not None: out.append({'scenario_family_id':key[0],'master_seed':key[1],'baseline_metric':bv,'comparison_metric':cv,'difference':cv-bv})
    s=stats([p['difference'] for p in out]); s['paired_sample_count']=s.pop('sample_count'); s['mean_paired_difference']=s.pop('mean',None); return {'pairs':out,'summary':s}
