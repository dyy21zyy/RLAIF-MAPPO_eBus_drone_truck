from tests.readiness_test_utils import run_pilot, load
def test_metric_sources(tmp_path):
 out=run_pilot(tmp_path); r=load(out,'metric_source_report.json'); assert r['passed']; names={m['metric'] for m in r['metrics']}; assert 'released_parcels' in names and 'runtime' in names
