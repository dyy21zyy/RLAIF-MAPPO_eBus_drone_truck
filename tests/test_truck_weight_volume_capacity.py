from tests.test_truck_batch_generation import env,p
from envs.action_generators.truck_batch_actions import generate_truck_batch_candidates
from types import SimpleNamespace

def cand(parcels, **cfg):
    e=env(parcels); e.config["truck"].update(cfg); t=SimpleNamespace(truck_id="t",current_location_id="depot_01",onboard_parcels=[],available_time=0); return generate_truck_batch_candidates(e,t)
def test_weight_volume_and_batch_limits_enforced():
    assert all(c.total_weight_kg<=2 or c.idle_flag for c in cand([p("p1",w=2),p("p2",w=2)], weight_capacity_kg=2))
    assert all(c.total_volume_m3<=2 or c.idle_flag for c in cand([p("p1",v=2),p("p2",v=2)], volume_capacity_m3=2))
    assert all(len(c.parcel_ids)<=1 for c in cand([p("p1"),p("p2")], max_batch_size=1))
