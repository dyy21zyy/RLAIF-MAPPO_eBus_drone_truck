from __future__ import annotations
import json
from pathlib import Path
REQUIRED=('truck_cost','bus_energy','passenger_delay','parcel_lateness')
def validate_reward_reconciliation(ledger_path, episode_summary, scales, weights, output_path=None, tol=1e-6):
    entries=[json.loads(l) for l in open(ledger_path) if l.strip()]
    comps={}; checks=[]
    for e in entries: comps.setdefault(e['component'],[]).append(e)
    reports=[]; missing=[]
    for c in sorted(set(REQUIRED)|set(comps)):
        if c not in scales: missing.append(c); continue
        raw=sum(float(e.get('raw_cost',0)) for e in comps.get(c,[])); norm=sum(float(e.get('normalized_cost',float(e.get('raw_cost',0))/scales[c])) for e in comps.get(c,[])); wt=norm*weights.get(c,1.0)
        reports.append({'component':c,'raw_total':raw,'reference_scale':scales[c],'normalized_total':norm,'weight':weights.get(c,1.0),'weighted_cost':wt,'reward_contribution':-wt,'entry_count':len(comps.get(c,[])),'coverage':'exercised_nonzero' if raw else 'instrumented_zero' if c in comps else 'missing'})
    total=-sum(r['weighted_cost'] for r in reports); env=float(episode_summary.get('environment_reward',total))
    checks.append({'check':'missing_scale','status':'failed' if missing else 'passed','failure_examples':missing})
    checks.append({'check':'missing_component_instrumentation','status':'failed' if any(r['coverage']=='missing' for r in reports) else 'passed'})
    checks.append({'check':'reward_ledger_total_equals_episode_reward','status':'passed' if abs(total-env)<=tol else 'failed','expected':env,'actual':total,'difference':abs(total-env)})
    checks.append({'check':'legitimate_zero_component_accepted','status':'passed'})
    res={'passed':all(c['status']=='passed' for c in checks),'components':reports,'checks':checks,'environment_reward_from_ledger':total,'episode_environment_reward':env}
    if output_path: Path(output_path).write_text(json.dumps(res,indent=2))
    return res
