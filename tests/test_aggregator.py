import json
from evaluation.aggregator import aggregate_directory, aggregate_records

def test_aggregator_mean_std_and_counts(tmp_path):
    rows=[{"method_name":"a","seed":0,"status":"success","episode_reward":1,"delivered_parcels":2}, {"method_name":"a","seed":1,"status":"success","episode_reward":3,"delivered_parcels":4}, {"method_name":"a","seed":2,"status":"skipped_missing_checkpoint"}]
    metrics,status=aggregate_records(rows); reward=next(r for r in metrics if r["metric"]=="episode_reward")
    assert reward["mean"]==2 and reward["std"]>0 and reward["number_of_seeds"]==3; assert status[0]["successful_runs"]==2 and status[0]["skipped_runs"]==1
    raw=tmp_path/"raw"; raw.mkdir();
    for index,row in enumerate(rows): (raw/f"{index}.json").write_text(json.dumps(row))
    aggregate_directory(raw,tmp_path/"summary"); assert (tmp_path/"summary"/"summary_metrics.csv").is_file()
