from __future__ import annotations
import argparse, json
from pathlib import Path
from utils.config import load_config
from training.config_resolver import resolve_mappo_training_config, validate_reward_artifacts

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

def build_training_seed_matrix(config, *, config_only=True):
    seeds=config.get('training',{}).get('training_seeds', [config.get('training',{}).get('seed')])
    runs=[resolve_mappo_training_config(config, seed_override=int(seed)) for seed in seeds]
    seen={}
    for r in runs:
        for key in ('checkpoint_path','training_log_path','eval_path','resolved_config_path'):
            path=r['output'][key]
            if path in seen: raise ValueError(f'duplicate output path across seeds: {path}')
            seen[path]=r['training']['seed']
        validate_reward_artifacts(r, config_only=config_only)
    return {'runs': [{'seed': r['training']['seed'], **r['output']} for r in runs]}

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,default=Path('configs/paper/benchmark.yaml')); ap.add_argument('--validate-only',action='store_true'); ap.add_argument('--run',action='store_true'); args=ap.parse_args(argv)
    cfg=load_config(args.config)
    if 'training' in cfg:
        matrix=build_training_seed_matrix(cfg, config_only=args.validate_only or not args.run); print(json.dumps(matrix, indent=2)); return 0
    validate_policy_matrix(cfg.get('policy_matrix',cfg)); print('policy matrix valid'); return 0
if __name__=='__main__': raise SystemExit(main())
