"""CLI for the isolated MAPPO policy post-training pipeline."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from training.post_training.config import resolve_post_training_config

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/post_training"))
    parser.add_argument("--validate-only", action="store_true"); parser.add_argument("--config-only", action="store_true")
    args = parser.parse_args(); config = resolve_post_training_config(args.config, seed=args.seed)
    if args.config_only:
        print(json.dumps(config, indent=2, sort_keys=True)); return
    if args.validate_only:
        print("post-training configuration and artifact paths are valid"); return
    raise RuntimeError("Full episode orchestration is intentionally unavailable in phase-one core; use --validate-only")

if __name__ == "__main__": main()
