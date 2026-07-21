from types import SimpleNamespace
from envs.status import is_delivered_status
from training.mappo_trainer import _episode_summary

def test_status_normalizes_and_summary_counts_delivered():
    assert is_delivered_status('delivered') and is_delivered_status('DELIVERED')
    env=SimpleNamespace(parcels={'a':SimpleNamespace(status='DELIVERED',delivered_time_min=5,deadline_min=7),'b':SimpleNamespace(status='WAITING',delivered_time_min=None,deadline_min=7)}, decision_counts={'assignment':0,'truck':0,'bus':0,'station':0}, cost_components={}, infeasible_action_corrections=0)
    s=_episode_summary(env, env_reward=1.0, rlaif_reward=0.0, bus_charging_count=0)
    assert s['delivered_parcels']==1 and s['undelivered_parcels']==1
