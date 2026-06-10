"""Evaluate deterministic Stage 7 asynchronous MAPPO policies."""
from __future__ import annotations
import argparse, importlib.util
from pathlib import Path
from utils.config import load_config

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, required=True); parser.add_argument('--checkpoint', type=Path, required=True); args=parser.parse_args(argv)
    if importlib.util.find_spec('torch') is None:
        print('SKIP: Stage 7 asynchronous MAPPO evaluation requires PyTorch.'); return 0
    from training.mappo_trainer import evaluate_mappo_async
    result=evaluate_mappo_async(load_config(args.config), args.checkpoint); print(result['summary']); return 0
if __name__=='__main__': raise SystemExit(main())
