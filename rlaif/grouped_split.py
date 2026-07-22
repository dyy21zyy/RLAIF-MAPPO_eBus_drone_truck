from __future__ import annotations
import hashlib,json,random,subprocess
from pathlib import Path
from typing import Any, Sequence

def group_key(r:Any, group_by:str='state')->tuple[str,...]:
    get=(lambda k: getattr(r,k) if hasattr(r,k) else r[k])
    if group_by=='state': return (str(get('scenario_id')),str(get('episode_id')),str(get('state_id')))
    if group_by=='episode': return (str(get('scenario_id')),str(get('episode_id')))
    if group_by=='scenario': return (str(get('scenario_id')),)
    raise ValueError('group_by must be state, episode, or scenario')
def manifest_hash(payload:Any)->str: return hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()
def _validate_fracs(a,b,c):
    if min(a,b,c)<0 or abs((a+b+c)-1.0)>1e-6: raise ValueError('split fractions must be nonnegative and sum to 1')
def grouped_split(records:Sequence[Any], train_ratio=.8, val_ratio=.1, test_ratio=.1, seed=42, group_by:str='state')->dict[str,Any]:
    _validate_fracs(train_ratio,val_ratio,test_ratio)
    groups=sorted({group_key(r,group_by) for r in records}); random.Random(seed).shuffle(groups); n=len(groups)
    n_train=int(round(n*train_ratio)); n_val=int(round(n*val_ratio));
    if n_train+n_val>n: n_val=max(0,n-n_train)
    splits={"train":groups[:n_train],"validation":groups[n_train:n_train+n_val],"test":groups[n_train+n_val:]}
    if n and not splits['train']: splits['train']=[splits['test'].pop()]
    rev={g:k for k,gs in splits.items() for g in gs}
    out={"group_by":group_by,"seed":seed,"splits":{k:[list(g) for g in gs] for k,gs in splits.items()},"records":{"train":[],"validation":[],"test":[]}}
    for r in records: out['records'][rev[group_key(r,group_by)]].append(r)
    validate_no_leakage(out); out['hash']=manifest_hash(out['splits']); return out
def validate_no_leakage(split:dict[str,Any])->None:
    sets={k:{tuple(x) for x in split['splits'][k]} for k in ('train','validation','test')}
    if sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['validation']&sets['test']: raise ValueError('split leakage detected')
def save_split_manifest(path:str|Path, split:dict[str,Any], **extra:Any)->None:
    payload={"manifest_version":1,"grouping_mode":split.get('group_by','state'),"split_seed":split.get('seed'),"train_group_ids":split['splits']['train'],"validation_group_ids":split['splits']['validation'],"test_group_ids":split['splits']['test'],"pair_counts_by_split":{k:len(v) for k,v in split['records'].items()},"split_hash":split['hash'],**extra}
    Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(payload,indent=2,sort_keys=True))
