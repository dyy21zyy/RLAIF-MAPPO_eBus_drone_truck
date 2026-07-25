"""Paired, policy-seed-first aggregation for fixed-policy robustness."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
import yaml

def rankdata(values):
    """Dependency-free average ranks (the only ranking needed here)."""
    x=np.asarray(values,float); order=np.argsort(x,kind="mergesort"); ranks=np.empty(len(x),float); i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]: j+=1
        ranks[order[i:j]]=(i+1+j)/2.0; i=j
    return ranks

def _wilcoxon_pvalue(d):
    """Two-sided signed-rank p-value with deterministic normal approximation."""
    d=np.asarray(d,float); d=d[d!=0]; n=len(d); ranks=rankdata(abs(d)); w=min(ranks[d>0].sum(),ranks[d<0].sum())
    mean=n*(n+1)/4; _,counts=np.unique(abs(d),return_counts=True); variance=(n*(n+1)*(2*n+1)-sum(c*(c+1)*(2*c+1) for c in counts if c>1))/24
    if variance<=0: return 1.0
    z=(abs(w-mean)-.5)/math.sqrt(variance); return min(1.0,math.erfc(max(0,z)/math.sqrt(2)))

LINEAGE={"policy_seed","paired_scenario_index","parameter_value","base_parameter_value","fallback_count","runtime_seconds"}

def rank_biserial(differences):
    d=np.asarray(differences,dtype=float); d=d[d!=0]
    if not len(d): return 0.0
    ranks=rankdata(abs(d)); return float((ranks[d>0].sum()-ranks[d<0].sum())/ranks.sum())

def paired_test(values, baseline):
    d=np.asarray(values,dtype=float)-np.asarray(baseline,dtype=float)
    if np.all(d==0): return {"p_value":1.0,"rank_biserial":0.0,"all_zero_differences":True}
    return {"p_value":float(_wilcoxon_pvalue(d)),"rank_biserial":rank_biserial(d),"all_zero_differences":False}

def holm_adjust(pvalues):
    p=np.asarray(pvalues,float); order=np.argsort(p); adjusted=np.empty(len(p)); running=0.0
    for rank,idx in enumerate(order): running=max(running,(len(p)-rank)*p[idx]); adjusted[idx]=min(1.0,running)
    return adjusted.tolist()

def bootstrap_ci(values, *, seed, resamples=10000, confidence=.95):
    x=np.asarray(values,float); rng=np.random.default_rng(seed); means=rng.choice(x,(resamples,len(x)),replace=True).mean(axis=1); alpha=(1-confidence)/2
    return tuple(float(v) for v in np.quantile(means,[alpha,1-alpha]))

def validate_rows(rows, *, expected_scenarios=100, expected_seeds=(1,2,3), formal=True):
    if not rows: raise ValueError("no episode rows")
    identities=[(r["sensitivity_family"],str(r["parameter_value"]),str(r["policy_seed"]),str(r["paired_scenario_index"]),r["scenario_id"],r["scenario_content_hash"]) for r in rows]
    if len(identities)!=len(set(identities)): raise ValueError("duplicate rows")
    if {r["sensitivity_mode"] for r in rows}!={"fixed_policy_robustness"}: raise ValueError("fixed/retrained sensitivity mixing")
    classifications={r["run_classification"] for r in rows}
    if len(classifications)!=1 or (formal and classifications!={"formal"}): raise ValueError("diagnostic/formal mixing")
    if any(r["status"]!="success" for r in rows): raise ValueError("failed rows cannot be aggregated")
    if any(int(float(r["fallback_count"]))!=0 for r in rows): raise ValueError("nonzero fallback count")
    groups={}
    for r in rows: groups.setdefault((r["sensitivity_family"],str(r["parameter_value"])),[]).append(r)
    scenario_sets={}
    for key,g in groups.items():
        if {int(r["policy_seed"]) for r in g}!=set(expected_seeds): raise ValueError("missing policy seeds")
        indices={int(r["paired_scenario_index"]) for r in g}
        if len(indices)!=expected_scenarios: raise ValueError("missing paired scenarios")
        for seed in expected_seeds:
            hs={r["policy_checkpoint_hash"] for r in g if int(r["policy_seed"])==seed}
            if len(hs)!=1: raise ValueError("mixed checkpoint hashes")
        if any(len({r["scenario_bank_hash"] for r in g if int(r["policy_seed"])==seed})!=1 for seed in expected_seeds): raise ValueError("mixed scenario-bank hashes")
        scenario_sets.setdefault(key[0],[]).append(indices)
    for sets in scenario_sets.values():
        if any(s!=sets[0] for s in sets[1:]): raise ValueError("mismatched paired scenario sets across levels")
    return groups

def aggregate_rows(rows, config, *, formal=True):
    expected=int(config["paired_scenarios"]["count"] if formal else len({r["paired_scenario_index"] for r in rows})); validate_rows(rows,expected_scenarios=expected,formal=formal)
    excluded=set(rows[0])-set(config["families"][rows[0]["sensitivity_family"]].get("primary_metrics",[]))
    metrics=sorted(set().union(*(set(config["families"][f].get("primary_metrics",[])) for f in config["families"])))
    paired=[]
    for (fam,val,idx),g in __import__("itertools").groupby(sorted(rows,key=lambda r:(r["sensitivity_family"],float(r["parameter_value"]),int(r["paired_scenario_index"]))),key=lambda r:(r["sensitivity_family"],float(r["parameter_value"]),int(r["paired_scenario_index"]))):
        gg=list(g); rec={"sensitivity_family":fam,"parameter_value":val,"paired_scenario_index":idx,"raw_successful_run_count":len(gg)}
        for metric in config["families"][fam]["primary_metrics"]:
            vals=[float(r[metric]) for r in gg]
            if any(not math.isfinite(x) for x in vals): raise ValueError(f"non-finite required metric: {metric}")
            rec[metric]=sum(vals)/len(vals)
        paired.append(rec)
    summaries=[]; tests=[]; ac=config["aggregation"]
    for fam,spec in config["families"].items():
      famrows=[r for r in paired if r["sensitivity_family"]==fam]; baseline=float(spec["baseline_value"])
      for metric in spec["primary_metrics"]:
        by={v:sorted([r for r in famrows if float(r["parameter_value"])==float(v)],key=lambda r:r["paired_scenario_index"]) for v in spec["values"]}; base=[r[metric] for r in by[baseline]]
        metric_tests=[]
        for val,recs in by.items():
            vals=[r[metric] for r in recs]; lo,hi=bootstrap_ci(vals,seed=int(ac["bootstrap_seed"]),resamples=int(ac["bootstrap_resamples"]),confidence=float(ac["confidence_level"]))
            summaries.append({"sensitivity_family":fam,"parameter_value":val,"metric":metric,"mean":float(np.mean(vals)),"standard_deviation":float(np.std(vals,ddof=1)),"median":float(np.median(vals)),"confidence_low":lo,"confidence_high":hi,"paired_scenario_count":len(vals),"raw_successful_run_count":3*len(vals),"failure_count":0})
            if float(val)!=baseline: metric_tests.append({"sensitivity_family":fam,"parameter_value":val,"baseline_value":baseline,"metric":metric,**paired_test(vals,base),"paired_scenario_count":len(vals)})
        adjusted=holm_adjust([x["p_value"] for x in metric_tests])
        for rec,adj in zip(metric_tests,adjusted): rec["holm_adjusted_p_value"]=adj
        tests.extend(metric_tests)
    return paired,summaries,tests

def _write_csv(path,rows):
    lineage=("sensitivity_family","parameter_value","paired_scenario_index","raw_successful_run_count")
    keys=set().union(*(row.keys() for row in rows)) if rows else set()
    fields=[field for field in lineage if field in keys]+sorted(keys-set(lineage)); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def run(config_path, output_root, *, validate_only=False):
    cfg=yaml.safe_load(Path(config_path).read_text()); root=Path(output_root); rows=[]
    for p in sorted((root/"evaluation").glob("**/episodes.csv")): rows.extend(csv.DictReader(p.open()))
    paired,summaries,tests=aggregate_rows(rows,cfg,formal=cfg["run_classification"]=="formal")
    if validate_only: return {"rows":len(rows),"status":"valid"}
    out=root/"summary"; _write_csv(out/"paired_scenario_means.csv",paired)
    for fam in cfg["families"]: _write_csv(out/f"{fam}_summary.csv",[r for r in summaries if r["sensitivity_family"]==fam]); _write_csv(out/f"{fam}_paired_tests.csv",[r for r in tests if r["sensitivity_family"]==fam])
    summary={"sensitivity_mode":"fixed_policy_robustness","aggregation_order":"policy seeds first, paired scenarios second","summary":summaries,"paired_tests":tests}; (out/"sensitivity_summary.json").write_text(json.dumps(summary,indent=2)+"\n"); (out/"aggregation_manifest.json").write_text(json.dumps({"status":"success","input_rows":len(rows),"bootstrap_seed":cfg["aggregation"]["bootstrap_seed"]},indent=2)+"\n"); return summary

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/paper/fixed_policy_sensitivity.yaml"); p.add_argument("--output-root",default="results/formal/sensitivity_fixed"); p.add_argument("--validate-only",action="store_true"); a=p.parse_args(argv); print(json.dumps(run(a.config,a.output_root,validate_only=a.validate_only),indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
