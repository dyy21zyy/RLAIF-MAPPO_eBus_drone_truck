import json
from pathlib import Path
from evaluation.scenario_bank import sha256_file
from experiments.resolve_formal_benchmark import sha256_file as bench_sha


def test_manifest_bank_hash_can_differ_from_file_hash(tmp_path):
    p=tmp_path/'scenario_bank_manifest.json'
    p.write_text(json.dumps({'split':'test','scenario_count':100,'bank_hash':'canonical-bank-hash','scenarios':[]})+'\n')
    assert sha256_file(p) != 'canonical-bank-hash'
