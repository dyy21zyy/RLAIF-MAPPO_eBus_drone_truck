from tests.readiness_test_utils import run_pilot, load
def test_checkpoint_roundtrip(tmp_path):
 out=run_pilot(tmp_path); r=load(out,'checkpoint_roundtrip_report.json'); assert r['passed']; s=r['action_snapshots'][0]; assert s['selected_action_before_save']==s['selected_action_after_load']; assert s['logits_before_save']==s['logits_after_load']; assert s['value_before_save']==s['value_after_load']
