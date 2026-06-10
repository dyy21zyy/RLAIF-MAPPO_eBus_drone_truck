"""Train Stage 7 asynchronous MAPPO."""
from __future__ import annotations
import argparse, importlib.util
from pathlib import Path
from utils.config import load_config

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, required=True); args=parser.parse_args(argv)
    if importlib.util.find_spec('torch') is None:
        print('SKIP: Stage 7 asynchronous MAPPO training requires PyTorch.'); return 0
    from training.mappo_trainer import train_mappo_async
    result=train_mappo_async(load_config(args.config)); print(f"Saved Stage 7 checkpoint to {result['checkpoint_path']}"); return 0
if __name__=='__main__': raise SystemExit(main())
