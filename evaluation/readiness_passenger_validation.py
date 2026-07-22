from __future__ import annotations
import csv,json
from pathlib import Path

def validate_passenger_reconciliation(path, output_path=None, tol=1e-6):
    rows=list(csv.DictReader(open(path,newline=''))); sums={}
    for k in ['waiting_increment','onboard_loading_delay','onboard_unloading_delay','onboard_charging_delay','normal_dwell_delay','post_dwell_past_delay','arrivals','boardings','alightings','residual_queue']:
        sums[k]=sum(float(r.get(k,0) or 0) for r in rows)
    final_wait=float(rows[-1].get('final_waiting_passenger_minutes',sums['waiting_increment'])) if rows else 0
    final_on=float(rows[-1].get('final_onboard_additional_delay',sums['onboard_loading_delay']+sums['onboard_unloading_delay']+sums['onboard_charging_delay'])) if rows else 0
    onboard=sums['onboard_loading_delay']+sums['onboard_unloading_delay']+sums['onboard_charging_delay']
    checks=[]
    def add(n,ok,exp,act): checks.append({'check':n,'status':'passed' if ok else 'failed','expected':exp,'actual':act,'absolute_difference':abs(exp-act),'relative_difference':abs(exp-act)/max(1,abs(exp))})
    add('waiting_entries_equal_final',abs(sums['waiting_increment']-final_wait)<=tol,final_wait,sums['waiting_increment'])
    add('onboard_components_equal_final',abs(onboard-final_on)<=tol,final_on,onboard)
    add('normal_dwell_excluded',abs(sums['normal_dwell_delay'])<=tol,0,sums['normal_dwell_delay'])
    add('new_passengers_no_past_delay',abs(sums['post_dwell_past_delay'])<=tol,0,sums['post_dwell_past_delay'])
    add('passenger_counts_reconcile',abs((sums['arrivals']-sums['boardings'])-sums['residual_queue'])<=tol,sums['arrivals']-sums['boardings'],sums['residual_queue'])
    res={'passed':all(c['status']=='passed' for c in checks),'checks':checks,'waiting_passenger_minutes':final_wait,'onboard_additional_delay_passenger_minutes':final_on}
    if output_path: Path(output_path).write_text(json.dumps(res,indent=2))
    return res
