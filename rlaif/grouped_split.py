from __future__ import annotations
import hashlib,json,random
from pathlib import Path
from typing import Any

def group_key(r:dict[str,Any])->tuple[str,str,str]: return (str(r["scenario_id"]),str(r["episode_id"]),str(r["state_id"]))
def manifest_hash(payload:Any)->str: return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
def grouped_split(records:list[dict[str,Any]], train_ratio=.8, val_ratio=.1, test_ratio=.1, seed=42)->dict[str,Any]:
    groups=sorted({group_key(r) for r in records}); random.Random(seed).shuffle(groups); n=len(groups)
    n_val=max(1,int(n*val_ratio)) if n>=3 else (1 if n==2 else 0); n_test=max(1,int(n*test_ratio)) if n>=3 else 0; n_train=max(0,n-n_val-n_test)
    splits={"train":groups[:n_train],"validation":groups[n_train:n_train+n_val],"test":groups[n_train+n_val:]}
    rev={g:k for k,gs in splits.items() for g in gs}
    out={"splits":{k:[list(g) for g in gs] for k,gs in splits.items()},"records":{"train":[],"validation":[],"test":[]}}
    for r in records: out["records"][rev[group_key(r)]].append(r)
    out["hash"]=manifest_hash(out["splits"]); return out

def save_split_manifest(path:str|Path, split:dict[str,Any])->None:
    Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps({"splits":split["splits"],"hash":split["hash"]},indent=2,sort_keys=True))
