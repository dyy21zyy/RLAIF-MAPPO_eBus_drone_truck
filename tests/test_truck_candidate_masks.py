from tests.test_truck_batch_generation import env,p
from envs.state_builder import build_truck_decision_surface
from types import SimpleNamespace

def test_idle_always_available_and_features_present():
    e=env([p("p1"),p("p2")]); e.trucks=[]; t=SimpleNamespace(truck_id="t",current_location_id="depot_01",onboard_parcels=[],available_time=0,status="idle",remaining_capacity_kg=10)
    e.trucks=[t]
    s=build_truck_decision_surface(e,t)
    assert any(a["idle_flag"]==1.0 for a in [c.features for c in s.candidates])
    assert all(s.action_mask())
    assert "waiting_parcel_count" in s.feature_names
