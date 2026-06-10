"""Run fair seed-controlled Stage 8 method comparisons."""
from __future__ import annotations
import argparse, json, tempfile
from pathlib import Path
from data_pipeline.build_instance import build_instance
from evaluation.aggregator import aggregate_directory
from evaluation.runner import EvaluationRunner
from evaluation.reporting import write_records_csv
from utils.config import load_config

def resolve_instance(config, *, temporary_root=None):
    exp=config["experiment"]
    explicit=exp.get("instance_path")
    if explicit and Path(explicit).is_file(): return Path(explicit)
    base=Path(exp.get("base_config","configs/shanghai_small.yaml"))
    output_root=temporary_root or exp.get("instance_output_root")
    instance=build_instance(base,fallback=bool(exp.get("fallback",True)),output_root=output_root)
    return Path(instance["output_directory"])/"instance.json"

def run_config(config, *, temporary_root=None):
    exp=dict(config["experiment"]); instance=resolve_instance(config,temporary_root=temporary_root)
    seeds=[int(s) for s in exp.get("seeds",[0])]
    limit=exp.get("max_episodes_per_method"); seeds=seeds[:int(limit)] if limit else seeds
    all_rows=[]
    for method in config.get("methods",[]):
        rows=EvaluationRunner(exp,instance,method,seeds[0] if seeds else 0).run_many(seeds); all_rows.extend(rows)
        if exp.get("fail_fast") and any(r["status"]=="failed" for r in rows): break
    write_records_csv(Path(exp["output_dir"])/"episodes.csv",all_rows)
    aggregate_directory(Path(exp["output_dir"])/"raw",Path(exp["output_dir"])/"summary")
    return all_rows

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,required=True); args=parser.parse_args(argv)
    rows=run_config(load_config(args.config)); print(json.dumps({"runs":len(rows),"statuses":{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})}},indent=2)); return 1 if any(r["status"]=="failed" for r in rows) else 0
if __name__=="__main__": raise SystemExit(main())
