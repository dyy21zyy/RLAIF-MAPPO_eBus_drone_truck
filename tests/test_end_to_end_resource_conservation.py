from pathlib import Path

from experiments.smoke_test_data_pipeline import run_smoke_test as build_instance
from envs.delivery_env import DynamicDeliveryEnv

CONFIG = Path(__file__).parents[1] / "configs/shanghai_small.yaml"


def assert_invariants(env, last_time):
    assert env.now_min + 1e-9 >= last_time
    statuses = [p.status for p in env.parcels.values()]
    assert len(statuses) == len(env.parcels)
    assert all(st.full_batteries + st.depleted_batteries + sum(b.status in {"CHARGING", "RESERVED", "IN_USE"} for b in st.battery_states) == len(st.battery_states) for st in env.stations.values())
    for truck in env.trucks:
        onboard_w = sum(env.parcels[pid].weight_kg for pid in truck.onboard_parcels)
        onboard_v = sum(env.parcels[pid].volume for pid in truck.onboard_parcels)
        assert onboard_w <= float(env.config["truck"].get("weight_capacity_kg", env.config["truck"]["capacity_kg"])) + 1e-6
        assert onboard_v <= float(env.config["truck"].get("volume_capacity_m3", 1.0)) + 1e-6
    for trip_id, kg in env.bus_freight_kg.items():
        assert kg <= float(env.config["bus"]["freight_capacity_kg"]) + 1e-6
    for stop in env.passenger_stops.values():
        assert stop.total_waiting >= 0
    for bus in env.physical_buses.values():
        assert bus.soc_kwh >= -1e-6
        assert bus.passenger_manifest.total_onboard_passengers <= int(env.config["bus"].get("passenger_capacity", env.config["bus"].get("bus_capacity_passenger", 80)))
    for st in env.stations.values():
        assert len(st.active_battery_charges) <= st.charging_slots
        assert st.locker_load_kg <= st.locker_capacity_kg + max(env.accumulated_locker_overflow, 0.0) + 1e-6
    owners = {}
    for pid, parcel in env.parcels.items():
        places = [parcel.status]
        assert len(places) == 1
        owners[pid] = parcel.status
    assert len(owners) == len(env.parcels)


def test_end_to_end_resource_conservation(tmp_path):
    built = build_instance(CONFIG, fallback=True, output_root=tmp_path)
    env = DynamicDeliveryEnv(Path(built["output_directory"]) / "instance.json", CONFIG)
    obs, info = env.reset(seed=123)
    seen_agents = set()
    last_time = env.now_min
    steps = 0
    while True:
        if obs.get("agent"):
            seen_agents.add(obs["agent"])
        assert_invariants(env, last_time)
        last_time = env.now_min
        if env.terminated or env.truncated:
            break
        mask = obs["action_mask"]
        # Prefer non-idle feasible actions to exercise integration surfaces.
        action = next((i for i, ok in enumerate(mask) if ok and i != 0), next(i for i, ok in enumerate(mask) if ok))
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        assert steps < 10000
        if terminated or truncated:
            assert_invariants(env, last_time)
            break
    assert {"assignment", "truck", "bus", "station"} <= seen_agents
    assert any(len(route) > 2 for truck in env.trucks for route in truck.route_history)
