from __future__ import annotations
import argparse, yaml
from pathlib import Path
from utils.config import load_config
from training.config_resolver import resolve_mappo_training_config, validate_reward_artifacts

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--config', type=Path, action='append', required=True); ap.add_argument('--config-only', action='store_true'); ns=ap.parse_args(argv)
    for c in ns.config:
        r=resolve_mappo_training_config(load_config(c)); validate_reward_artifacts(r, config_only=ns.config_only); print(yaml.safe_dump(r, sort_keys=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
