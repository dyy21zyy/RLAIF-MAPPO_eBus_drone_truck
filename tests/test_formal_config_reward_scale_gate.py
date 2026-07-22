import json, hashlib, pytest
from training.config_resolver import validate_run_classification, TrainingConfigError


def cfg(path='missing.json', h='a'*64):
    return {'run_classification':'formal','env':{'fallback':False},'reward':{'apply_reference_scales':True,'scale_artifact':path,'scale_artifact_hash':h}}


def test_missing_path_fails_validate_only():
    with pytest.raises(TrainingConfigError, match='existing artifact'):
        validate_run_classification(cfg(), config_only=False)


def test_missing_and_placeholder_hash_fail():
    for h in [None, '', 'TBD', 'placeholder', 'freeze-after-estimation']:
        with pytest.raises(TrainingConfigError, match='scale_artifact_hash'):
            validate_run_classification(cfg(h=h), config_only=True)


def test_config_only_accepts_external_nonplaceholder_path():
    validate_run_classification(cfg(), config_only=True)


def test_hash_mismatch_fails(tmp_path):
    p=tmp_path/'s.json'; payload={'artifact_version':1,'scales':{'a':1}}
    payload['artifact_hash']=hashlib.sha256(json.dumps({'artifact_version':1,'scales':{'a':1}}, sort_keys=True).encode()).hexdigest()
    p.write_text(json.dumps(payload))
    with pytest.raises(TrainingConfigError, match='matching artifact hash'):
        validate_run_classification(cfg(str(p), 'b'*64), config_only=False)
