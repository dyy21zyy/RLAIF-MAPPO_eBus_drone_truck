from tests.readiness_test_utils import run_pilot, load
def test_all_mappo_agents_and_critic_update(tmp_path):
 out=run_pilot(tmp_path); r=load(out,'mappo_update_report.json'); assert r['passed'];
 for a in ['assignment','truck','bus','station']: assert r['agents'][a]['transition_count']>0 and r['agents'][a]['changed_parameter_count']>0
 assert r['critic']['changed_parameter_count']>0 and r['gae_finite'] and r['returns_finite']
