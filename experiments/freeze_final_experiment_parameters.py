"""Build or validate immutable final experiment parameter freeze artifacts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import yaml
from evaluation.parameter_freeze import load_freeze_template, validate_freeze_template, git_metadata, canonical_hash
from evaluation.parameter_provenance import build_parameter_freeze_report

def _file_hash(path:Path)->str|None:
    return canonical_hash({'path':str(path),'content':path.read_text(encoding='utf-8')}) if path.is_file() else None

def build_artifact(config:dict, config_path:str)->dict:
    report=validate_freeze_template(config)
    contract=Path('configs/paper/method_difference_contract.yaml')
    artifact={
        'freeze_schema_version':config.get('freeze_schema_version'),
        **git_metadata(),
        'resolved_parameters':build_parameter_freeze_report(config, source_file=config_path)['parameters'],
        'method_difference_contract_hash':_file_hash(contract),
        'source_config_hashes':{config_path:_file_hash(Path(config_path)), str(contract):_file_hash(contract)},
        'scenario_bank_hashes':{k:config['scenario_protocol'][k]['value'] for k in ('train_bank_hash','validation_bank_hash','test_bank_hash')},
        'reward_scale_hash':config['reward_reference_scale']['artifact_hash']['value'],
        'reward_checkpoint_hashes':{a:v.get('checkpoint_hash') for a,v in config['rlaif_parameters']['methods']['mappo_rlaif_all']['agents'].items()},
        'validation_report':report,
    }
    artifact['freeze_hash']=canonical_hash({k:v for k,v in artifact.items() if k!='freeze_hash'})
    return artifact

def main(argv=None)->int:
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True); p.add_argument('--output',required=True)
    p.add_argument('--validate-only',action='store_true'); p.add_argument('--report-only',action='store_true'); p.add_argument('--force',action='store_true')
    a=p.parse_args(argv)
    cfg=load_freeze_template(a.config); report=validate_freeze_template(cfg)
    print(report['status'])
    if report['unresolved_placeholders']:
        print('Unresolved formal artifact hashes:')
        for item in report['unresolved_placeholders']: print(f"- {item['path']}: {item['value']} -> {item['blocked_status']}")
    if a.validate_only: return 0
    out=Path(a.output)
    if out.exists() and not a.force:
        print(f'existing freeze artifact requires --force: {out}', file=sys.stderr); return 2
    payload=build_parameter_freeze_report(cfg, source_file=a.config) if a.report_only else build_artifact(cfg,a.config)
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')
    return 0
if __name__=='__main__': raise SystemExit(main())
