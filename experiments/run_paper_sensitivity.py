from __future__ import annotations
MODES={'fixed_policy_robustness','retrained_policy_sensitivity'}
def validate_sensitivity(config):
    modes={e.get('mode') for e in config.get('experiments',[]) if e.get('mode')}
    if modes and not modes<=MODES: raise ValueError(f'unsupported sensitivity modes: {modes-MODES}')
    if len(modes)>1 and any(e.get('combined_table') for e in config.get('experiments',[])): raise ValueError('fixed-policy and retrained-policy sensitivity must remain separate')
    for e in config.get('experiments',[]):
        mode=e.get('mode')
        if mode=='fixed_policy_robustness' and not e.get('policy_checkpoint_hash', e.get('fixed_policy_checkpoint_hash')) and e.get('strict_artifacts'):
            raise ValueError('fixed-policy mode must preserve checkpoint hash')
        if mode=='retrained_policy_sensitivity':
            for d in e.get('dimensions',[]):
                if ('value' in d or 'values' in d) and not d.get('checkpoint') and not d.get('policy_checkpoint'):
                    raise ValueError('retrained sensitivity requires a parameter-specific checkpoint')
    return True
def sensitivity_row(mode, parameter_name, parameter_value, policy_checkpoint_hash, scenario_bank_hash):
    return {'sensitivity_mode':mode,'parameter_name':parameter_name,'parameter_value':parameter_value,'policy_checkpoint_hash':policy_checkpoint_hash,'scenario_bank_hash':scenario_bank_hash}
def main(argv=None):
    import argparse, yaml
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); a=p.parse_args(argv); validate_sensitivity(yaml.safe_load(open(a.config))); print('sensitivity config valid'); return 0
if __name__=='__main__': raise SystemExit(main())
