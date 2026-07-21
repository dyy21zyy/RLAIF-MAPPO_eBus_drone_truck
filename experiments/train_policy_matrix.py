from __future__ import annotations
import argparse,json
from pathlib import Path
from utils.config import load_config
REQUIRED_POLICIES=("mappo_env","mappo_rlaif_assignment","mappo_rlaif_all")
def validate_policy_matrix(config):
    names={p['name'] for p in config.get('policies',[])}
    missing=[p for p in REQUIRED_POLICIES if p not in names]
    if missing: raise ValueError(f'missing required policies: {missing}')
    seen={}
    for p in config.get('policies',[]):
        ck=p.get('checkpoint')
        if not ck: raise ValueError(f"policy {p.get('name')} missing checkpoint")
        if ck in seen and p.get('name')!=seen[ck]: raise ValueError('policy comparisons require separate checkpoints')
        seen[ck]=p.get('name')
        if p.get('rlaif_enabled') and not p.get('reward_checkpoints'): raise ValueError('RLAIF policy requires reward checkpoints used during training')
    return True
def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,default=Path('configs/paper/benchmark.yaml')); ap.add_argument('--validate-only',action='store_true'); args=ap.parse_args(argv); validate_policy_matrix(load_config(args.config).get('policy_matrix',load_config(args.config))); print('policy matrix valid'); return 0
if __name__=='__main__': raise SystemExit(main())
