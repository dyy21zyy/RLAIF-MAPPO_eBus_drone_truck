"""Prespecified paired analysis for the assignment hard-locker experiment."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np

METHODS = ("mappo_env", "mappo_rlaif_assignment")
SEEDS = (1, 2, 3)
PRIMARY_METRICS = ("fulfillment_rate","environment_reward","truck_distance","drone_missions","bus_charging_energy","bus_propulsion_energy","truck_volume_utilization","truck_weight_utilization")
SECONDARY_METRICS = ("infeasible_action_count","overload_kw_min","overload_duration","runtime")
FEASIBILITY_METRICS = ("locker_overflow_amount","locker_overflow_duration","total_locker_reserved_kg","reward_fallback_count","invariant_failure_count")
ALIASES = {
 "environment_reward": ("environment_reward","env_reward","total_environment_reward"),
 "truck_distance": ("truck_distance","truck_distance_km"), "drone_missions": ("drone_missions","drone_mission_count"),
 "bus_charging_energy": ("bus_charging_energy","bus_charging_energy_kwh"), "bus_propulsion_energy": ("bus_propulsion_energy","bus_propulsion_energy_kwh"),
 "overload_duration": ("overload_duration","overload_duration_min"), "runtime": ("runtime","runtime_seconds"),
 "locker_overflow_amount": ("locker_overflow_amount","locker_overflow_kg"), "locker_overflow_duration": ("locker_overflow_duration","locker_overflow_duration_min"),
 "total_locker_reserved_kg": ("total_locker_reserved_kg","terminal_locker_reserved_kg"), "reward_fallback_count": ("reward_fallback_count","fallback_count"),
 "invariant_failure_count": ("invariant_failure_count","invariant_failures"),
}


def sha256_file(path: Path) -> str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()


def unwrap_formal_metric(row: dict[str,Any], metric_name: str) -> float:
 formal=row.get("formal_metrics")
 if not isinstance(formal,dict): raise ValueError("formal_metrics must be a mapping")
 for alias in ALIASES.get(metric_name,(metric_name,)):
  if alias in formal:
   record=formal[alias]
   available = record.get("available", record.get("availability") == "available") if isinstance(record,dict) else False
   if not isinstance(record,dict) or available is not True or "value" not in record: raise ValueError(f"malformed or unavailable formal metric {metric_name}")
   value=float(record["value"])
   if not math.isfinite(value): raise ValueError(f"non-finite {metric_name}")
   return value
 raise ValueError(f"required formal metric {metric_name} is missing")

def _metric(row: dict[str,Any], name: str) -> float:
 formal=row.get("formal_metrics")
 if formal is not None:
  return unwrap_formal_metric(row,name)
 sources=(row,row.get("metrics",{}),row.get("episode_metrics",{}),row.get("summary",{}))
 for alias in ALIASES.get(name,(name,)):
  for source in sources:
   if alias in source:
    value=float(source[alias])
    if not math.isfinite(value): raise ValueError(f"non-finite {name}")
    return value
 raise ValueError(f"required metric {name} is missing")


def _is_hash(value: Any) -> bool:
 return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdefABCDEF' for c in value)


def load_and_validate(path: Path) -> tuple[list[dict[str,Any]], list[str]]:
 if not path.is_file() or not path.stat().st_size: raise ValueError("benchmark episode JSONL is missing or empty")
 rows=[]
 for n,line in enumerate(path.read_text().splitlines(),1):
  if line.strip():
   try: rows.append(json.loads(line))
   except json.JSONDecodeError as exc: raise ValueError(f"invalid JSONL row {n}") from exc
 if len(rows)!=600: raise ValueError(f"expected exactly 600 rows, found {len(rows)}")
 if {r.get('method_id') for r in rows} != set(METHODS): raise ValueError("expected exactly the two learned methods")
 if {int(r.get('training_seed',-1)) for r in rows} != set(SEEDS): raise ValueError("training seeds must be exactly 1, 2, and 3")
 failed=[r for r in rows if r.get('status','success') not in ('success','complete','passed') or r.get('failed') is True]
 if failed: raise ValueError(f"benchmark contains {len(failed)} failed rows")
 scenarios=sorted({str(r.get('scenario_id')) for r in rows})
 if len(scenarios)!=100: raise ValueError(f"expected 100 common scenarios, found {len(scenarios)}")
 seen=set(); scenario_hashes={}
 for row in rows:
  key=(row['method_id'],int(row['training_seed']),str(row['scenario_id']))
  if key in seen: raise ValueError(f"duplicate method/seed/scenario row: {key}")
  seen.add(key)
  sh=row.get('scenario_content_hash')
  if not _is_hash(sh): raise ValueError(f"invalid scenario content hash for {key}")
  if key[2] in scenario_hashes and scenario_hashes[key[2]]!=sh: raise ValueError(f"scenario content hash mismatch for {key[2]}")
  scenario_hashes[key[2]]=sh
  if not _is_hash(row.get('policy_checkpoint_hash')): raise ValueError(f"invalid checkpoint hash for {key}")
  artifacts=row.get('artifact_hashes',{})
  reward_hashes=row.get('reward_checkpoint_hashes',{})
  scale_hash=row.get('reward_scale_artifact_hash')
  if not _is_hash(scale_hash): raise ValueError(f"invalid reward-scale artifact hash for {key}")
  if row['method_id']=='mappo_rlaif_assignment' and not _is_hash(reward_hashes.get('assignment') or artifacts.get('assignment_reward_model')):
   raise ValueError(f"invalid assignment reward artifact hash for {key}")
  for metric in PRIMARY_METRICS+SECONDARY_METRICS+FEASIBILITY_METRICS: _metric(row,metric)
 expected={(m,s,scenario) for m in METHODS for s in SEEDS for scenario in scenarios}
 missing=expected-seen
 if missing: raise ValueError(f"missing method/seed/scenario row: {next(iter(missing))}")
 for metric in FEASIBILITY_METRICS:
  nonzero=[r for r in rows if _metric(r,metric)!=0.0]
  if nonzero: raise ValueError(f"feasibility metric {metric} must be structurally zero")
 return rows,scenarios


def scenario_seed_means(rows: list[dict[str,Any]], scenarios: list[str]) -> list[dict[str,Any]]:
 by={(r['method_id'],int(r['training_seed']),str(r['scenario_id'])):r for r in rows}; output=[]
 for scenario in scenarios:
  item={'scenario_id':scenario}
  for method in METHODS:
   for metric in PRIMARY_METRICS+SECONDARY_METRICS+FEASIBILITY_METRICS:
    item[f'{method}__{metric}']=float(np.mean([_metric(by[(method,seed,scenario)],metric) for seed in SEEDS]))
  output.append(item)
 return output


def paired_bootstrap_ci(differences: np.ndarray, repetitions: int, seed: int) -> tuple[float,float]:
 """Resample paired scenario indices and return percentile CI of mean differences."""
 rng=np.random.default_rng(seed); means=[]
 for start in range(0,repetitions,2000):
  count=min(2000,repetitions-start); indices=rng.integers(0,len(differences),size=(count,len(differences)))
  means.append(differences[indices].mean(axis=1))
 values=np.concatenate(means)
 return float(np.quantile(values,.025)),float(np.quantile(values,.975))


def rank_biserial(differences: np.ndarray) -> float:
 nonzero=differences[differences!=0]
 if not len(nonzero): return 0.0
 ranks=_rankdata(np.abs(nonzero)); total=float(ranks.sum())
 return float((ranks[nonzero>0].sum()-ranks[nonzero<0].sum())/total)


def _rankdata(values: np.ndarray) -> np.ndarray:
 order=np.argsort(values,kind='mergesort'); ranks=np.empty(len(values),dtype=float); start=0
 while start<len(values):
  end=start+1
  while end<len(values) and values[order[end]]==values[order[start]]: end+=1
  ranks[order[start:end]]=(start+1+end)/2.0; start=end
 return ranks


def paired_wilcoxon_pvalue(differences: np.ndarray) -> float:
 """Two-sided Wilcoxon signed-rank p-value (normal approximation with tie correction)."""
 values=differences[differences!=0]
 if not len(values): return 1.0
 ranks=_rankdata(np.abs(values)); positive=float(ranks[values>0].sum()); n=len(values)
 mean=n*(n+1)/4.0
 _,counts=np.unique(np.abs(values),return_counts=True)
 variance=(n*(n+1)*(2*n+1)-sum(int(t)*(int(t)+1)*(2*int(t)+1) for t in counts if t>1))/24.0
 if variance<=0: return 1.0
 # Continuity correction toward the null mean.
 distance=max(0.0,abs(positive-mean)-0.5); z=distance/math.sqrt(variance)
 return min(1.0,math.erfc(z/math.sqrt(2.0)))


def holm_adjust(pvalues: list[float]) -> list[float]:
 order=np.argsort(pvalues); adjusted=[0.0]*len(pvalues); running=0.0; n=len(pvalues)
 for rank,index in enumerate(order):
  running=max(running,min(1.0,(n-rank)*pvalues[index])); adjusted[index]=running
 return adjusted


def calculate_statistics(means: list[dict[str,Any]], *, bootstrap_seed: int, bootstrap_repetitions: int) -> list[dict[str,Any]]:
 if bootstrap_repetitions<20000: raise ValueError("formal paired bootstrap requires at least 20,000 repetitions")
 output=[]
 for family,metrics in (("primary",PRIMARY_METRICS),("secondary",SECONDARY_METRICS)):
  family_rows=[]
  for offset,metric in enumerate(metrics):
   env=np.asarray([r[f'mappo_env__{metric}'] for r in means],dtype=float); rai=np.asarray([r[f'mappo_rlaif_assignment__{metric}'] for r in means],dtype=float); diff=rai-env
   zeros=int(np.count_nonzero(diff==0)); p=paired_wilcoxon_pvalue(diff)
   low,high=paired_bootstrap_ci(diff,bootstrap_repetitions,bootstrap_seed+offset+(0 if family=='primary' else 1000))
   family_rows.append({'family':family,'metric':metric,'mappo_env_mean':float(env.mean()),'mappo_rlaif_assignment_mean':float(rai.mean()),'mappo_env_std':float(env.std(ddof=1)),'mappo_rlaif_assignment_std':float(rai.std(ddof=1)),'mappo_env_median':float(np.median(env)),'mappo_rlaif_assignment_median':float(np.median(rai)),'paired_mean_difference':float(diff.mean()),'paired_median_difference':float(np.median(diff)),'bootstrap_ci_95_low':low,'bootstrap_ci_95_high':high,'wilcoxon_p_value':p,'holm_adjusted_p_value':None,'rank_biserial_effect_size':rank_biserial(diff),'scenario_count':len(diff),'zero_difference_count':zeros,'difference_direction':'mappo_rlaif_assignment - mappo_env'})
  adjusted=holm_adjust([r['wilcoxon_p_value'] for r in family_rows])
  for row,value in zip(family_rows,adjusted): row['holm_adjusted_p_value']=value
  output.extend(family_rows)
 return output


def _write_csv(path:Path,rows:list[dict[str,Any]])->None:
 if not rows: raise ValueError('cannot write empty CSV')
 with path.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def analyze(input_path: Path, output_dir: Path, *, bootstrap_seed: int=8001, bootstrap_repetitions: int=20000, code_commit: str|None=None) -> dict[str,Any]:
 rows,scenarios=load_and_validate(input_path); means=scenario_seed_means(rows,scenarios)
 stats=calculate_statistics(means,bootstrap_seed=bootstrap_seed,bootstrap_repetitions=bootstrap_repetitions)
 output_dir.mkdir(parents=True,exist_ok=True)
 means_path=output_dir/'scenario_seed_means.csv'; stats_path=output_dir/'paired_statistics.csv'; feasibility_path=output_dir/'feasibility_checks.json'; excluded_path=output_dir/'excluded_redundant_metrics.json'
 _write_csv(means_path,means); _write_csv(stats_path,stats)
 feasibility={'status':'pass','statistical_tests_performed':False,'metrics':{metric:{'structural_zero':True,'row_count':600,'nonzero_count':0} for metric in FEASIBILITY_METRICS}}
 feasibility_path.write_text(json.dumps(feasibility,indent=2,sort_keys=True)+'\n')
 excluded={'delivered_parcel_count':'excluded when release count is constant because it is redundant with fulfillment_rate','combined_rlaif_reward':'not a cross-method primary outcome','exact_duplicate_metrics':[]}
 # Document exact duplicates without excluding prespecified primary/secondary outcomes.
 vectors={metric:tuple(_metric(r,metric) for r in rows) for metric in PRIMARY_METRICS+SECONDARY_METRICS+FEASIBILITY_METRICS}
 for i,left in enumerate(vectors):
  for right in list(vectors)[i+1:]:
   if vectors[left]==vectors[right]: excluded['exact_duplicate_metrics'].append([left,right])
 excluded_path.write_text(json.dumps(excluded,indent=2,sort_keys=True)+'\n')
 outputs=[means_path,stats_path,feasibility_path,excluded_path]
 commit=code_commit or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 manifest={'input_path':str(input_path),'input_file_hash':sha256_file(input_path),'code_commit':commit,'scenario_count':100,'method_ids':list(METHODS),'training_seeds':list(SEEDS),'bootstrap_seed':bootstrap_seed,'bootstrap_repetitions':bootstrap_repetitions,'metric_families':{'primary':list(PRIMARY_METRICS),'secondary':list(SECONDARY_METRICS),'feasibility':list(FEASIBILITY_METRICS)},'statistical_unit':'scenario after averaging three training seeds within method','difference_direction':'mappo_rlaif_assignment - mappo_env','output_file_hashes':{p.name:sha256_file(p) for p in outputs}}
 (output_dir/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n'); return manifest


def main(argv: Iterable[str]|None=None)->int:
 p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--bootstrap-seed',type=int,default=8001); p.add_argument('--bootstrap-repetitions',type=int,default=20000); a=p.parse_args(argv)
 try: analyze(a.input,a.output_dir,bootstrap_seed=a.bootstrap_seed,bootstrap_repetitions=a.bootstrap_repetitions); return 0
 except Exception as exc: print(f'paired hard-locker analysis failed: {exc}',file=__import__('sys').stderr); return 2
if __name__=='__main__': raise SystemExit(main())
