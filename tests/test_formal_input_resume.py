import json, pytest
from pathlib import Path
from experiments.prepare_formal_experiment_inputs import _validate_banks, prepare


def _bank(root: Path, split: str, count: int, bank_hash='canonical'):
    d=root/'scenarios'/split; d.mkdir(parents=True, exist_ok=True)
    (d/'scenario_bank_manifest.json').write_text(json.dumps({'split':split,'bank':split,'scenario_count':count,'bank_hash':f'{bank_hash}-{split}','scenarios':[]})+'\n')
    return d


def test_resume_reuses_valid_empty_banks(tmp_path, monkeypatch):
    paths={s:_bank(tmp_path,s,c) for s,c in {'train':1,'validation':1,'test':1}.items()}
    manifests=_validate_banks(paths, {'train':1,'validation':1,'test':1})
    assert manifests['train']['bank_hash']=='canonical-train'


def test_resume_rejects_missing_bank(tmp_path):
    with pytest.raises(RuntimeError, match='missing scenario bank manifest'):
        prepare(tmp_path, resume=True, counts={'train':1,'validation':1,'test':1}, scale_scenario_limit=0)


def test_resume_force_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError, match='mutually exclusive'):
        prepare(tmp_path, resume=True, force=True, counts={'train':1,'validation':1,'test':1}, scale_scenario_limit=0)
