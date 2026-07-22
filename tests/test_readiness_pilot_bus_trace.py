import csv, json
from evaluation.readiness_event_validation import validate_event_chain
from tests.readiness_test_utils import run_pilot, load
def test_bus_trace_and_event_chain(tmp_path):
 out=run_pilot(tmp_path); rows=list(csv.DictReader(open(out/'bus_stop_trace.csv'))); r=load(out,'event_chain_validation.json'); assert r['passed']; assert any(x['stop_type']=='ordinary' for x in rows); assert any(x['stop_type']=='integrated' for x in rows); assert len({x['trip_id'] for x in rows})>=2
def test_duplicate_stop_arrivals_fail(tmp_path):
 out=run_pilot(tmp_path); rows=list(csv.DictReader(open(out/'bus_stop_trace.csv'))); rows.append(rows[0]); p=out/'bad.csv'; import csv as C; w=C.DictWriter(open(p,'w',newline=''),fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows); assert not validate_event_chain(p)['passed']
