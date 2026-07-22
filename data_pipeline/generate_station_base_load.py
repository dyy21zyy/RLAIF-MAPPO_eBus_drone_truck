from __future__ import annotations
import csv, json, hashlib
from pathlib import Path
from typing import Any
import numpy as np
FIELDS=["station_id","interval_id","start_min","end_min","base_load_kw","station_base_load_seed","source"]
def _sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def generate_station_base_load(stations: list[dict[str, Any]], config: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    scfg=config.get('station',{}); seed=int(config.get('seeds',{}).get('station_base_load_seed', config.get('project',{}).get('seed',0)+505))
    horizon=float(config.get('bus',{}).get('delivery_horizon_min', config.get('time',{}).get('delivery_evaluation_horizon_min',480)))
    step=float(scfg.get('base_load_interval_min',15.0)); lo=float(scfg.get('base_load_min_kw',80.0)); hi=float(scfg.get('base_load_max_kw',180.0))
    rng=np.random.default_rng(seed); rows=[]
    for st in sorted(stations, key=lambda x: str(x.get('station_id',x.get('id')))):
        sid=str(st.get('station_id',st.get('id'))); t=0.0; k=0
        while t < horizon-1e-9:
            e=min(horizon,t+step); rows.append({'station_id':sid,'interval_id':f'{sid}_{k:04d}','start_min':f'{t:.9f}','end_min':f'{e:.9f}','base_load_kw':f'{float(rng.uniform(lo,hi)):.9f}','station_base_load_seed':seed,'source':'project_extension_uniform_80_180_kw'}); t=e; k+=1
    csvp=out/'station_base_load.csv'
    with csvp.open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    prov=out/'station_base_load_provenance.json'; prov.write_text(json.dumps({'process':'piecewise_constant_seeded_uniform','station_base_load_seed':seed,'min_kw':lo,'max_kw':hi,'interval_min':step,'horizon_min':horizon,'station_count':len(stations),'interval_count':len(rows),'source_classification':'project_extension','artifact_hashes':{'station_base_load.csv':_sha(csvp)}},indent=2),encoding='utf-8')
    return {'station_base_load':csvp,'station_base_load_provenance':prov}
