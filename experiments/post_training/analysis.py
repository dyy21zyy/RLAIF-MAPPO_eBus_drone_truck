"""Pre-registered scenario-paired post-training comparisons."""
from __future__ import annotations
COMPARISONS=(
 {"comparison_id":"rlaif_post_vs_env_continued","treatment":"mappo_rlaif_assignment_post","baseline":"mappo_env_post_continued","inference":"confirmatory"},
 {"comparison_id":"env_continued_vs_base","treatment":"mappo_env_post_continued","baseline":"mappo_env_post_base","inference":"exploratory"},
 {"comparison_id":"rlaif_post_vs_base","treatment":"mappo_rlaif_assignment_post","baseline":"mappo_env_post_base","inference":"exploratory"},)
FEASIBILITY_METRICS={"stranded_passengers","invalid_actions","constraint_violations"}
def rank_biserial(diffs):
    nz=[x for x in diffs if x!=0]; return 0.0 if not nz else (sum(x>0 for x in nz)-sum(x<0 for x in nz))/len(nz)
def analyze(rows,metrics,bootstrap_samples=2000,seed=2026):
    import numpy as np
    from scipy.stats import wilcoxon
    grouped={}
    for r in rows:
      for metric in metrics: grouped.setdefault((r["method_id"],r["scenario_id"],metric),[]).append(float(r["formal_metrics"][metric]))
    rng=np.random.default_rng(seed); out=[]
    for comp in COMPARISONS:
      for metric in metrics:
       scenarios=sorted({k[1] for k in grouped if k[0]==comp["treatment"] and k[2]==metric})
       if len(scenarios)!=100: raise ValueError("statistical unit must be 100 common scenarios")
       d=np.array([np.mean(grouped[(comp["treatment"],s,metric)])-np.mean(grouped[(comp["baseline"],s,metric)]) for s in scenarios])
       item={**comp,"metric":metric,"metric_family":metric.split(".")[0],"n_scenarios":100,"mean_difference":float(d.mean()),"rank_biserial":rank_biserial(d)}
       if metric in FEASIBILITY_METRICS: item.update(test="not_applicable_feasibility",p_value=None,ci95=None)
       else:
        boot=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(bootstrap_samples)])
        item.update(test="paired_wilcoxon",p_value=float(wilcoxon(d).pvalue) if np.any(d) else 1.0,ci95=[float(x) for x in np.quantile(boot,[.025,.975])])
       out.append(item)
    # Holm correction is scoped to comparison × metric family.
    for comp in COMPARISONS:
      families={x["metric_family"] for x in out if x["comparison_id"]==comp["comparison_id"]}
      for fam in families:
       items=[x for x in out if x["comparison_id"]==comp["comparison_id"] and x["metric_family"]==fam and x["p_value"] is not None]
       ordered=sorted(items,key=lambda x:x["p_value"]); running=0.0
       for i,x in enumerate(ordered): running=max(running,min(1.0,x["p_value"]*(len(items)-i))); x["holm_p_value"]=running
    return out
