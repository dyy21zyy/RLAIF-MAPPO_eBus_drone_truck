"""Train or validate asynchronous MAPPO configs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
from rlaif.torch_runtime import is_torch_runtime_available
from utils.config import load_config
from training.config_resolver import resolve_mappo_training_config, validate_reward_artifacts

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--config-only', action='store_true')
    parser.add_argument('--output-root', type=Path)
    args=parser.parse_args(argv)
    resolved=resolve_mappo_training_config(load_config(args.config), seed_override=args.seed, output_root_override=args.output_root)
    validate_reward_artifacts(resolved, config_only=args.config_only)
    if args.validate_only:
        print(yaml.safe_dump(resolved, sort_keys=False)); return 0
    Path(resolved['output']['resolved_config_path']).parent.mkdir(parents=True, exist_ok=True)
    Path(resolved['output']['resolved_config_path']).write_text(yaml.safe_dump(resolved, sort_keys=False), encoding='utf-8')
    if not is_torch_runtime_available():
        print('SKIP: Stage 7 asynchronous MAPPO training requires PyTorch.'); return 0
    from training.mappo_trainer import train_mappo_async
    result=train_mappo_async(resolved); print(f"Saved Phase 7 checkpoint to {result['checkpoint_path']}"); return 0
if __name__=='__main__': raise SystemExit(main())
