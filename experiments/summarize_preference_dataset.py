from __future__ import annotations
import argparse,json,hashlib,subprocess
from pathlib import Path
import yaml
from rlaif.reward_model_dataset import build_reward_pair_dataset,dataset_hash
from rlaif.grouped_split import grouped_split, save_split_manifest

def load_jsonl(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--preferences',required=True); ap.add_argument('--config',required=True); ap.add_argument('--output',required=True); ns=ap.parse_args()
    cfg=yaml.safe_load(Path(ns.config).read_text()); formal=cfg.get('run_classification','formal')=='formal'; agent=cfg['agent_type']
    rows=load_jsonl(ns.preferences); ds=build_reward_pair_dataset(rows,agent_type=agent,formal_mode=formal,require_bus_event_coverage=(formal and agent=='bus'))
    if formal and len(ds)==0: raise SystemExit('formal validation failed: no usable binary data')
    spcfg=cfg.get('split',{}); split=grouped_split(ds.examples, spcfg.get('train_fraction',.7), spcfg.get('validation_fraction',.15), spcfg.get('test_fraction',.15), spcfg.get('seed',1), spcfg.get('group_by','scenario'))
    report=ds.report.to_dict(); report.update({'split_leakage_status':'pass','minimum_data_gate_status':'pass' if len(ds)>=(cfg.get('training',{}).get('min_samples',0)) else 'fail','bus_event_coverage': sorted(set(report['counts_by_event']) & {'BUS_TERMINAL_DEPARTURE','BUS_STATION_ARRIVAL'}),'preference_file_hash':sha(ns.preferences),'resolved_dataset_hash':dataset_hash(ds),'split_counts':{k:len(v) for k,v in split['records'].items()}})
    out=Path(ns.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,sort_keys=True))
    save_split_manifest(out.with_name('split_manifest.json'), split, preference_file_hash=report['preference_file_hash'], resolved_dataset_hash=report['resolved_dataset_hash'], counts_by_agent=report['counts_by_agent'], counts_by_event=report['counts_by_event'], counts_by_label_source=report['counts_by_label_source'], counts_by_outcome=report['counts_by_outcome'])
if __name__=='__main__': main()
