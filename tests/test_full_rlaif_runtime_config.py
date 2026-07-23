import json, yaml
from pathlib import Path
from experiments.prepare_formal_rlaif_artifacts import _inject_artifacts
from training.event_schema import AGENT_TYPES


def manifest(tmp_path):
    return {'scenario_bank_hash':'bankhash','reward_scale_hash':'scalehash','agents':{a:{'checkpoint_path':str(tmp_path/f'reward_{a}.pt'),'checkpoint_hash':f'hash_{a}','supported_event_types':[]} for a in AGENT_TYPES}}


def test_complete_assignment_and_full_configs_generated_without_placeholders(tmp_path):
    cfg={'scenario_bank':{'final_train_manifest':str(tmp_path/'bank.json')}}; (tmp_path/'bank.json').write_text('{"split":"train","scenario_count":1,"bank_hash":"bankhash","scenarios":[]}')
    out1=tmp_path/'configs'/'mappo_rlaif_assignment.yaml'; out2=tmp_path/'configs'/'mappo_rlaif_all.yaml'
    _inject_artifacts(Path('configs/paper/train_mappo_rlaif_assignment.yaml'), out1, manifest(tmp_path), ('assignment',), cfg)
    _inject_artifacts(Path('configs/paper/train_mappo_rlaif_all.yaml'), out2, manifest(tmp_path), tuple(AGENT_TYPES), cfg)
    for p in (out1,out2):
        text=p.read_text(); assert not any(x in text for x in ['REPLACE_WITH','PLACEHOLDER','MISSING_FORMAL','TBD','UNKNOWN'])
        data=yaml.safe_load(text)
        assert {'run_classification','mode','env','training','networks','reward','rlaif','output'} <= set(data)
        assert data['rlaif']['fallback_to_env_reward'] is False
        assert data['rlaif']['fail_on_invalid_reward_model'] is True
    a=yaml.safe_load(out1.read_text()); f=yaml.safe_load(out2.read_text())
    assert [k for k,v in a['rlaif']['agents'].items() if v.get('enabled')] == ['assignment']
    assert [k for k,v in f['rlaif']['agents'].items() if v.get('enabled')] == ['assignment','truck','bus','station']


def test_mappo_method_fairness_except_rlaif_scope_and_outputs(tmp_path):
    env=yaml.safe_load(Path('configs/paper/train_mappo_env.yaml').read_text())
    cfg={'scenario_bank':{'final_train_manifest':str(tmp_path/'bank.json')}}; (tmp_path/'bank.json').write_text('{"split":"train","scenario_count":1,"bank_hash":"bankhash","scenarios":[]}')
    out=tmp_path/'configs'/'mappo_rlaif_all.yaml'; _inject_artifacts(Path('configs/paper/train_mappo_rlaif_all.yaml'), out, manifest(tmp_path), tuple(AGENT_TYPES), cfg)
    r=yaml.safe_load(out.read_text())
    for key in ('training','networks','reward'):
        if key=='reward':
            e={k:v for k,v in env[key].items() if 'hash' not in k and 'expected_training' not in k}
            rr={k:v for k,v in r[key].items() if 'hash' not in k and 'expected_training' not in k}
            assert e==rr
        else:
            e={k:v for k,v in env[key].items() if k!='training_seeds'}
            rr={k:v for k,v in r[key].items() if k!='training_seeds'}
            assert e==rr
