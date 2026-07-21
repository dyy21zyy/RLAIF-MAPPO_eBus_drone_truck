from evaluation.statistics import summarize_metric, bootstrap_ci

def test_skipped_runs_excluded_from_statistics():
    rows=[{"status":"success","m":10},{"status":"skipped_missing_checkpoint","m":0},{"status":"failed","m":0}]
    s=summarize_metric(rows,"m",seed=1)
    assert s['mean']==10 and s['skip_count']==1 and s['failure_count']==1

def test_bootstrap_results_are_reproducible():
    assert bootstrap_ci([1,2,3],seed=7)==bootstrap_ci([1,2,3],seed=7)
