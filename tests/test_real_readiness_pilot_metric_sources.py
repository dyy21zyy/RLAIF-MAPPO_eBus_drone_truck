from tests.real_readiness_helpers import run_pilot, load

def test_metric_sources(tmp_path):
    r=load(run_pilot(tmp_path),"metric_source_report.json")
    assert r["passed"] and any(m["metric"]=="released_parcels" for m in r["metrics"]) and all(v["availability"] for v in r["metrics"] )
