from __future__ import annotations
import argparse,json
from pathlib import Path
from evaluation.aggregator import load_records
from evaluation.statistics import summarize_metric, paired_difference
from evaluation.paired_evaluation import group_by_method

def aggregate(input_dir, metric='total_normalized_cost'):
    rows=load_records(input_dir); grouped=group_by_method(rows)
    return {m:summarize_metric(rs,metric,seed=0) for m,rs in grouped.items()}
def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); ap.add_argument('--metric',default='total_normalized_cost'); args=ap.parse_args(argv)
    out=aggregate(args.input,args.metric); Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2,sort_keys=True)); print(f'wrote {args.output}'); return 0
if __name__=='__main__': raise SystemExit(main())
