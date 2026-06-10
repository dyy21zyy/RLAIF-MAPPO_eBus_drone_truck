"""Run only dependency-light baselines from a Stage 8 benchmark config."""
import argparse
from pathlib import Path
from experiments.run_benchmark import run_config
from utils.config import load_config

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,default=Path("configs/experiments.yaml")); args=parser.parse_args(argv)
    config=load_config(args.config); config["methods"]=[m for m in config.get("methods",[]) if m.get("assignment_policy") not in {"assignment_ppo","mappo_async"}]
    rows=run_config(config); print(f"Completed {len(rows)} baseline episodes."); return 1 if any(r["status"]=="failed" for r in rows) else 0
if __name__=="__main__": raise SystemExit(main())
