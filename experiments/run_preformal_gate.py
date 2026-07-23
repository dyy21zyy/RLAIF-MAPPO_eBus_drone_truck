from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.preformal_gate import run_config

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--config', required=True)
    a=p.parse_args(argv); report=run_config(a.config); print(json.dumps({'overall_status':report['overall_status']},indent=2));
    return 0 if report['overall_status'] in {'PREFORMAL_DIAGNOSTIC_PASSED','PREFORMAL_ALL_REQUIRED_PATHS_PASSED','PREFORMAL_ASSIGNMENT_RLAIF_PASSED_FULL_RLAIF_BLOCKED','PREFORMAL_ENVIRONMENT_PATH_PASSED_RLAIF_BLOCKED'} else 1
if __name__=='__main__': raise SystemExit(main())
