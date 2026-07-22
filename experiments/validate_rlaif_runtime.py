from __future__ import annotations
import argparse, sys, yaml
from pathlib import Path
from rlaif.reward_registry import RewardRegistry

PLACEHOLDER='REPLACE_WITH_REAL_HASH'

def validate(config_path: str, strict: bool=False) -> str:
    cfg=yaml.safe_load(Path(config_path).read_text())
    r=cfg.get('rlaif',{})
    if any(c.get('checkpoint_hash') == PLACEHOLDER for c in r.get('agents',{}).values() if c.get('enabled')):
        return 'BLOCKED_PLACEHOLDER_HASH'
    for c in r.get('agents',{}).values():
        if c.get('enabled') and not Path(str(c.get('checkpoint'))).is_file():
            return 'BLOCKED_MISSING_REWARD_CHECKPOINT'
    try:
        RewardRegistry(cfg)
    except Exception as exc:
        msg=str(exc)
        if 'formal loading requires run_classification' in msg: return 'BLOCKED_SMOKE_CHECKPOINT_IN_FORMAL_MODE'
        return 'BLOCKED_CHECKPOINT_INCOMPATIBLE'
    return 'READY_FORMAL_RLAIF' if cfg.get('run_classification') == 'formal' else 'READY_DIAGNOSTIC_RLAIF'

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--report-only',action='store_true'); ap.add_argument('--strict',action='store_true')
    ns=ap.parse_args(argv); status=validate(ns.config, ns.strict); print(status); return 0 if (ns.report_only or status.startswith('READY')) else 1
if __name__=='__main__': sys.exit(main())
