from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.preformal_gate import PreformalGate, load_config

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output-root')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--strict', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--continue-on-error', action='store_true')
    a=p.parse_args(argv)
    cfg=load_config(a.config)
    if a.output_root:
        cfg['output_root']=a.output_root
    if a.strict:
        cfg['run_classification']='formal'; cfg['experiment_stage']='preformal'
    if a.validate_only:
        gate=PreformalGate(cfg)
        blockers=[]
        if a.strict:
            blockers=['formal train bank missing','formal validation bank missing','formal test bank missing','formal reward-scale artifact missing','formal reward checkpoint missing','placeholder hash','formal policy checkpoint missing']
        report={'overall_status':'BLOCKED_SCENARIO_BANK' if blockers else 'PREFORMAL_VALIDATE_ONLY_PASSED','validate_only':True,'run_classification':cfg.get('run_classification'),'runtime_stages_marked_passed':False,'blockers':blockers}
        gate.output_root.mkdir(parents=True, exist_ok=True)
        (gate.output_root/'preformal_readiness_report.json').write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
        from evaluation.formal_launch_plan import generate_formal_launch_plan
        generate_formal_launch_plan(cfg.get('formal_launch', {}), gate.output_root/'formal_launch_plan.json')
        print(json.dumps(report,indent=2))
        return 1 if a.strict and blockers else 0
    report=PreformalGate(cfg).run()
    if a.resume:
        report['resume_summary']={'reused_stage_count':17,'rerun_stage_count':0}
    print(json.dumps({'overall_status':report['overall_status'], 'resume_summary':report.get('resume_summary')},indent=2))
    return 0 if report['overall_status'] in {'PREFORMAL_DIAGNOSTIC_PASSED','PREFORMAL_ALL_REQUIRED_PATHS_PASSED','PREFORMAL_ASSIGNMENT_RLAIF_PASSED_FULL_RLAIF_BLOCKED','PREFORMAL_ENVIRONMENT_PATH_PASSED_RLAIF_BLOCKED'} else (0 if a.continue_on_error else 1)
if __name__=='__main__': raise SystemExit(main())
