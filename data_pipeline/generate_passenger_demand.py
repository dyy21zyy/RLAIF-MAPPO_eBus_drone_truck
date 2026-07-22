"""Generate deterministic time-dependent passenger demand artifacts."""
from __future__ import annotations
import csv, json, hashlib
from pathlib import Path
from typing import Any
from collections import Counter
from envs.dynamics.passenger_demand import blocks_from_config, generate_time_dependent_arrivals, sample_truncated_normal_rates

RATE_FIELDS=["stop_id","baseline_rate_per_min","baseline_rate_seed","baseline_distribution","baseline_mean","baseline_std","baseline_min","baseline_max","source"]
TEMPORAL_FIELDS=["block_id","start_min","end_min","multiplier","source"]
EVENT_FIELDS=["passenger_event_id","origin_stop_id","destination_stop_id","arrival_time_min","passenger_count","block_id","baseline_rate_per_min","demand_intensity","temporal_multiplier","effective_rate_per_min","passenger_seed"]
def _sha(p: Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()

def generate_passenger_demand(stops: list[dict[str, Any]], config: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output=Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    stop_ids=[str(s.get("stop_id", s.get("id"))) for s in stops]
    pcfg=config.setdefault("passenger", {})
    seeds=config.get("seeds",{})
    arr_seed=int(seeds.get("passenger_seed", config.get("project",{}).get("seed",0)))
    baseline_seed=int(seeds.get("passenger_baseline_rate_seed", arr_seed+104729))
    intensity=float(pcfg.get("passenger_demand_intensity", pcfg.get("demand_intensity",1.0)))
    horizon=float(config.get("bus",{}).get("delivery_horizon_min", config.get("time",{}).get("delivery_evaluation_horizon_min",480)))
    mean=float(pcfg.get("baseline_rate_mean_per_min",0.25)); std=float(pcfg.get("baseline_rate_std_per_min",0.10)); lo=float(pcfg.get("baseline_rate_min_per_min",0.05)); hi=float(pcfg.get("baseline_rate_max_per_min",0.60))
    rates=sample_truncated_normal_rates(stop_ids, seed=baseline_seed, mean=mean, std=std, min_rate=lo, max_rate=hi)
    blocks=blocks_from_config(pcfg, horizon)
    events=generate_time_dependent_arrivals(stop_ids,horizon_min=horizon,baseline_rates=rates,demand_intensity=intensity,temporal_blocks=blocks,seed=arr_seed)
    rates_path=output/"passenger_stop_rates.csv"
    with rates_path.open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=RATE_FIELDS); w.writeheader()
        for sid in sorted(rates):
            w.writerow({'stop_id':sid,'baseline_rate_per_min':f'{rates[sid]:.9f}','baseline_rate_seed':baseline_seed,'baseline_distribution':'truncated_normal','baseline_mean':mean,'baseline_std':std,'baseline_min':lo,'baseline_max':hi,'source':'paper_derived'})
    temporal_path=output/"passenger_temporal_profile.csv"
    with temporal_path.open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=TEMPORAL_FIELDS); w.writeheader()
        for b in blocks: w.writerow({'block_id':b.block_id,'start_min':f'{b.start_min:.9f}','end_min':f'{b.end_min:.9f}','multiplier':f'{b.multiplier:.9f}','source':'project_extension'})
    events_path=output/"passenger_arrivals.csv"
    with events_path.open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=EVENT_FIELDS); w.writeheader()
        for e in events:
            w.writerow({'passenger_event_id':e.passenger_event_id,'origin_stop_id':e.origin_stop_id,'destination_stop_id':e.destination_stop_id,'arrival_time_min':f'{e.arrival_time_min:.9f}','passenger_count':e.passenger_count,'block_id':e.block_id,'baseline_rate_per_min':f'{e.baseline_rate_per_min:.9f}','demand_intensity':f'{e.demand_intensity:.9f}','temporal_multiplier':f'{e.temporal_multiplier:.9f}','effective_rate_per_min':f'{e.effective_rate_per_min:.9f}','passenger_seed':e.passenger_seed})
    prov_path=output/"passenger_demand_provenance.json"
    by_block=Counter(e.block_id for e in events); by_stop=Counter(e.origin_stop_id for e in events)
    hashes={p.name:_sha(p) for p in (rates_path,temporal_path,events_path)}
    prov_path.write_text(json.dumps({'process':'piecewise_time_dependent_poisson','seed':arr_seed,'baseline_rate_seed':baseline_seed,'baseline_rate_distribution':{'mean':mean,'std':std,'min':lo,'max':hi,'source':'paper_derived'},'temporal_blocks':[b.__dict__ for b in blocks],'demand_intensity':intensity,'stop_count':len(stop_ids),'event_count':len(events),'event_count_by_block':dict(sorted(by_block.items())),'event_count_by_stop':dict(sorted(by_stop.items())),'artifact_hashes':hashes,'source_classification':{'baseline_rates':'paper_derived','temporal_profile':'project_extension'},'destination_rule':'downstream_uniform_explicit_destination'},indent=2,sort_keys=True),encoding='utf-8')
    return {'passenger_stop_rates':rates_path,'passenger_temporal_profile':temporal_path,'passenger_arrivals':events_path,'passenger_demand_provenance':prov_path}
