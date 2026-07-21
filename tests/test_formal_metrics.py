from types import SimpleNamespace
from evaluation.metrics import collect_formal_metrics, FORMAL_METRICS

def test_on_time_over_all_uses_all_released_parcels():
    env=SimpleNamespace(parcels={1:SimpleNamespace(status='DELIVERED',delivered_time_min=5,deadline_min=10,release_time_min=0,is_urgent=False),2:SimpleNamespace(status='PENDING',delivered_time_min=None,deadline_min=10,release_time_min=0,is_urgent=False)},trucks=[],stations={},bus_soc_kwh={},cost_components={},infeasible_action_corrections=0)
    m=collect_formal_metrics(env)
    assert m['on_time_over_all_released']==0.5 and m['on_time_over_delivered']==1.0

def test_urgent_metrics_use_explicit_urgent_status_not_priority():
    env=SimpleNamespace(parcels={1:SimpleNamespace(status='DELIVERED',delivered_time_min=5,deadline_min=10,release_time_min=0,is_urgent=False,priority=9),2:SimpleNamespace(status='DELIVERED',delivered_time_min=5,deadline_min=10,release_time_min=0,is_urgent=True,priority=1)},trucks=[],stations={},bus_soc_kwh={},cost_components={},infeasible_action_corrections=0)
    m=collect_formal_metrics(env)
    assert m['urgent_on_time_fulfillment']==1.0
    assert set(FORMAL_METRICS) <= set(m)
