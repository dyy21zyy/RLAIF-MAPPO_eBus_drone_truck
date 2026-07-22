"""Readiness validation for diagnostic bus/event traces."""
from __future__ import annotations
import csv,json,math
from collections import defaultdict,Counter
from pathlib import Path
from training.event_schema import AUTOMATIC_EVENT_TYPES, decision_event_agent, is_decision_event

def _rows(p):
    with open(p,newline='') as f: return list(csv.DictReader(f))
def _check(name, ok, count=0, examples=None): return {"check":name,"status":"passed" if ok else "failed","evidence_count":count,"failure_examples":examples or []}
def validate_event_chain(bus_trace_path, event_trace_path=None, output_path=None, min_layover=5.0):
    rows=_rows(bus_trace_path); checks=[]
    checks.append(_check('ordinary_stops_present', any(r.get('stop_type')=='ordinary' for r in rows), sum(r.get('stop_type')=='ordinary' for r in rows)))
    checks.append(_check('integrated_stations_present', any(r.get('stop_type')=='integrated' for r in rows), sum(r.get('stop_type')=='integrated' for r in rows)))
    bad=[]
    for r in rows:
        for k in ('actual_arrival','actual_departure','soc_at_arrival','soc_at_departure'):
            try: float(r[k])
            except Exception: bad.append({k:r.get(k)})
    checks.append(_check('soc_is_finite', not bad, len(rows), bad[:3]))
    bytrip=defaultdict(list)
    for r in rows: bytrip[(r['physical_bus_id'],r['trip_id'])].append(r)
    order_bad=[]; dup_bad=[]; causal_bad=[]; seg_bad=[]
    for key,rs in bytrip.items():
        rs=sorted(rs,key=lambda r:int(r['event_sequence']))
        idx=[int(r['stop_index']) for r in rs]
        if idx != sorted(idx): order_bad.append({str(key):idx})
        if len(idx)!=len(set(idx)): dup_bad.append({str(key):idx})
        for a,b in zip(rs,rs[1:]):
            if float(b['actual_arrival']) < float(a['actual_departure'])-1e-9: causal_bad.append({'previous':a,'next':b})
            if int(b['stop_index']) != int(a['stop_index'])+1: seg_bad.append({'previous':a['stop_index'],'next':b['stop_index']})
    checks += [_check('trip_visits_stops_in_order',not order_bad,len(bytrip),order_bad[:2]),_check('one_arrival_per_stop_no_duplicates',not dup_bad,len(rows),dup_bad[:2]),_check('downstream_arrival_after_preceding_departure',not causal_bad,len(rows),causal_bad[:2]),_check('segment_count_matches_route_progression',not seg_bad,len(rows),seg_bad[:2])]
    bybus=defaultdict(list)
    for r in rows: bybus[r['physical_bus_id']].append(r)
    soc_bad=[]; lay_bad=[]; shift=False
    for bus,rs in bybus.items():
        trips=defaultdict(list)
        for r in rs: trips[r['trip_id']].append(r)
        keys=sorted(trips, key=lambda t:min(int(r['event_sequence']) for r in trips[t]))
        for a,b in zip(keys,keys[1:]):
            last=sorted(trips[a],key=lambda r:int(r['stop_index']))[-1]; first=sorted(trips[b],key=lambda r:int(r['stop_index']))[0]
            if abs(float(last['soc_at_departure'])-float(first['soc_before_incoming_segment']))>1e-6: soc_bad.append({'bus':bus,'from':a,'to':b})
            if float(first['actual_departure']) < float(last['actual_departure'])+min_layover-1e-9: lay_bad.append({'bus':bus,'from':a,'to':b})
            if float(first['actual_departure']) > float(first['scheduled_departure'])+1e-9: shift=True
    checks += [_check('soc_persists_across_trips',not soc_bad,len(bybus),soc_bad[:2]),_check('minimum_layover_enforced',not lay_bad,len(bybus),lay_bad[:2]),_check('trip_one_delay_affects_trip_two',shift,len(bybus))]
    if event_trace_path and Path(event_trace_path).exists():
        events=[json.loads(l) for l in open(event_trace_path) if l.strip()]; times=[e.get('time',0) for e in events]
        checks.append(_check('event_times_nondecreasing', times==sorted(times), len(events)))
        stale=[e for e in events if e.get('event_type')=='BUS_ARRIVE_STOP' and e.get('stale')]
        checks.append(_check('no_stale_arrival_event', not stale, len(events), stale[:2]))
        bad_agents=[]; auto=[]
        for e in events:
            et=e.get('event_type'); ag=e.get('agent_type')
            if et in AUTOMATIC_EVENT_TYPES and e.get('creates_mappo_transition'): auto.append(e)
            if is_decision_event(et) and decision_event_agent(et)!=ag: bad_agents.append(e)
        checks += [_check('automatic_events_create_no_mappo_transitions',not auto,len(events),auto[:2]),_check('decision_events_belong_to_expected_agents',not bad_agents,len(events),bad_agents[:2])]
    passed=all(c['status']=='passed' for c in checks); out={'passed':passed,'checks':checks}
    if output_path: Path(output_path).write_text(json.dumps(out,indent=2))
    return out
