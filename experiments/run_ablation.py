"""Run configured Stage 8/Phase 9 ablations, skipping unavailable learned artifacts."""
import argparse
from pathlib import Path
from experiments.run_benchmark import run_config
from utils.config import load_config

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,required=True); args=parser.parse_args(argv)
    config=load_config(args.config); config["methods"]=config.pop("ablations",config.get("methods",[])); rows=run_config(config)
    print(f"Processed {len(rows)} ablation episodes (unavailable checkpoints are explicit skips)."); return 1 if any(r["status"]=="failed" for r in rows) else 0
if __name__=="__main__": raise SystemExit(main())

# Phase 9 paper wrappers in experiments/run_paper_* add strict validation.
