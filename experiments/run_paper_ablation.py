from __future__ import annotations
import hashlib
SUPPORTED_ABLATIONS={'no_rlaif','assignment_only_rlaif','full_rlaif','no_event_time_discount','no_event_embedding','no_centralized_critic','single_parcel_truck','truck_batching','greedy_bus_loading','learned_bus_loading','automatic_battery_charging','learned_battery_charging','no_passenger_aware_charging'}
def validate_ablations(config):
    seen={}
    for a in config.get('ablations',[]):
        name=a.get('name')
        if name not in SUPPORTED_ABLATIONS: raise ValueError(f'unsupported ablation: {name}')
        if not (a.get('config_difference') or a.get('config_switch')): raise ValueError(f'ablation {name} requires actual configuration difference')
        if a.get('requires_retraining', a.get('training_requirement') in ('retrain','separate_checkpoint')):
            ck=a.get('checkpoint') or a.get('checkpoint_path')
            h=a.get('checkpoint_hash') or (hashlib.sha256(str(ck).encode()).hexdigest() if ck else None)
            if not ck or not h: raise ValueError(f'ablation {name} requiring retraining needs checkpoint and hash')
            if h in seen: raise ValueError(f'duplicate checkpoint hash across retrained ablations: {name} and {seen[h]}')
            seen[h]=name
        if name=='no_event_embedding' and a.get('uses_event_embedding_checkpoint'): raise ValueError('no-event-embedding ablation cannot use a standard event-embedding checkpoint')
    return True

def main(argv=None):
    import argparse, yaml
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); a=p.parse_args(argv); validate_ablations(yaml.safe_load(open(a.config))); print('ablation config valid'); return 0
if __name__=='__main__': raise SystemExit(main())
