from __future__ import annotations
import csv,json
from pathlib import Path

def validate_station_power(path, output_path=None, tol=1e-6):
    rows=list(csv.DictReader(open(path,newline=''))); checks=[]; overload=dur=buskw=battkw=0; peak=max(float(r['total_load_kw']) for r in rows) if rows else 0; bad=[]
    boundaries=[]
    for r in rows:
        dt=float(r.get('duration_min',0)); total=float(r['total_load_kw']); comp=float(r['base_load_kw'])+float(r['bus_charging_load_kw'])+float(r['battery_charging_load_kw'])
        if abs(total-comp)>tol: bad.append(r)
        overload += max(0,total-float(r['capacity_kw']))*dt; dur += dt if total>float(r['capacity_kw']) else 0; buskw += float(r['bus_charging_load_kw'])*dt; battkw += float(r['battery_charging_load_kw'])*dt; boundaries.append(float(r['start_min']))
    summary=rows[-1] if rows else {}
    def add(n,ok,exp,act): checks.append({'check':n,'status':'passed' if ok else 'failed','expected':exp,'actual':act,'difference':abs(float(exp)-float(act)) if isinstance(exp,(int,float)) else None})
    add('total_load_equals_components',not bad,0,len(bad)); add('peak_load_matches_trace',abs(float(summary.get('peak_load_kw',peak))-peak)<=tol,float(summary.get('peak_load_kw',peak)),peak)
    add('overload_kw_min_exact',abs(float(summary.get('overload_kw_min',overload))-overload)<=tol,float(summary.get('overload_kw_min',overload)),overload); add('overload_duration_exact',abs(float(summary.get('overload_duration_min',dur))-dur)<=tol,float(summary.get('overload_duration_min',dur)),dur)
    add('bus_kw_min_to_kwh',abs(float(summary.get('bus_charging_energy_kwh',buskw/60))-buskw/60)<=tol,float(summary.get('bus_charging_energy_kwh',buskw/60)),buskw/60); add('battery_kw_min_to_kwh',abs(float(summary.get('battery_charging_energy_kwh',battkw/60))-battkw/60)<=tol,float(summary.get('battery_charging_energy_kwh',battkw/60)),battkw/60)
    add('integration_boundaries_complete',0 in boundaries and 480 in [float(r.get('end_min',-1)) for r in rows],True,True)
    res={'passed':all(c['status']=='passed' for c in checks),'checks':checks,'peak_load_kw':peak,'overload_kw_min':overload,'overload_duration_min':dur,'bus_charging_energy_kwh':buskw/60,'battery_charging_energy_kwh':battkw/60}
    if output_path: Path(output_path).write_text(json.dumps(res,indent=2))
    return res
