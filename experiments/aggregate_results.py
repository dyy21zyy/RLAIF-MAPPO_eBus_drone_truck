"""Aggregate Stage 8 raw episode records."""
import argparse
from evaluation.aggregator import aggregate_directory

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--input",required=True); parser.add_argument("--output",required=True); args=parser.parse_args(argv)
    metrics,statuses=aggregate_directory(args.input,args.output); print(f"Aggregated {len(statuses)} methods and {len(metrics)} metric rows into {args.output}"); return 0
if __name__=="__main__": raise SystemExit(main())
