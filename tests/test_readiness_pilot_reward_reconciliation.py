import json
from evaluation.readiness_reward_validation import validate_reward_reconciliation
from tests.readiness_test_utils import run_pilot, load
def test_reward_reconciliation(tmp_path):
 out=run_pilot(tmp_path); r=load(out,'reward_reconciliation.json'); assert r['passed']; assert r['episode_environment_reward']==r['environment_reward_from_ledger']
def test_missing_scale_fails(tmp_path):
 p=tmp_path/'l.jsonl'; p.write_text(json.dumps({'component':'truck_cost','raw_cost':1})+'\n'); assert not validate_reward_reconciliation(p,{}, {}, {})['passed']
