"""Evaluate deterministic Phase 7 environment-reward-only asynchronous MAPPO policies."""
from __future__ import annotations
import argparse
from pathlib import Path
from rlaif.torch_runtime import is_torch_runtime_available
from utils.config import load_config

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, required=True); parser.add_argument('--checkpoint', type=Path, required=True); args=parser.parse_args(argv)
    if not is_torch_runtime_available():
        print('SKIP: Stage 7 asynchronous MAPPO evaluation requires PyTorch.'); return 0
    from training.mappo_trainer import evaluate_mappo_async
    result=evaluate_mappo_async(load_config(args.config), args.checkpoint); print(result['summary']); return 0
if __name__=='__main__': raise SystemExit(main())
