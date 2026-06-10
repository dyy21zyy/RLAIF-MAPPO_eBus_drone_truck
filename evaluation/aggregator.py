"""Aggregate raw Stage 8 episode JSON records by method."""
from __future__ import annotations
import csv, json, math
from pathlib import Path
from statistics import fmean, median, stdev

KEY_METRICS=("delivered_parcels","undelivered_parcels","on_time_delivery_rate","average_parcel_lateness","late_delivery_count","truck_total_distance","passenger_delay","bus_operating_delay","bus_charging_energy","locker_overflow_amount","power_overload_amount","infeasible_action_count","episode_reward")

def load_records(input_dir):
    records=[]
    for path in sorted(Path(input_dir).rglob("*.json")):
        try: value=json.loads(path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError): continue
        if isinstance(value,dict) and "method_name" in value and "status" in value: records.append(value)
    return records

def aggregate_records(records):
    methods={}
    for row in records: methods.setdefault(str(row["method_name"]),[]).append(row)
    metrics=[]; statuses=[]
    for method, rows in sorted(methods.items()):
        successful=[r for r in rows if r.get("status")=="success"]
        statuses.append({"method_name":method,"number_of_seeds":len({r.get('seed') for r in rows}),"successful_runs":len(successful),"skipped_runs":sum(str(r.get('status','')).startswith('skipped') for r in rows),"failed_runs":sum(r.get('status')=="failed" for r in rows)})
        for metric in KEY_METRICS:
            values=[float(r[metric]) for r in successful if isinstance(r.get(metric),(int,float)) and math.isfinite(float(r[metric]))]
            metrics.append({"method_name":method,"metric":metric,"mean":fmean(values) if values else None,"std":stdev(values) if len(values)>1 else 0.0 if values else None,"min":min(values) if values else None,"max":max(values) if values else None,"median":median(values) if values else None,"number_of_seeds":len({r.get('seed') for r in rows}),"successful_runs":len(successful),"skipped_runs":sum(str(r.get('status','')).startswith('skipped') for r in rows),"failed_runs":sum(r.get('status')=="failed" for r in rows)})
    return metrics,statuses

def aggregate_directory(input_dir, output_dir):
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    metrics,statuses=aggregate_records(load_records(input_dir))
    for name, rows in (("summary_metrics.csv",metrics),("method_status.csv",statuses)):
        with (output/name).open("w",encoding="utf-8",newline="") as handle:
            fields=list(rows[0]) if rows else (["method_name","metric"] if "summary" in name else ["method_name"]); writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (output/"summary_metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return metrics,statuses
