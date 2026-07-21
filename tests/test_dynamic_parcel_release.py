from data_pipeline.build_instance import build_instance
from envs.delivery_env import DynamicDeliveryEnv

def test_reset_starts_unreleased_and_release_activates_assignment(tmp_path):
    inst=build_instance('configs/shanghai_small.yaml', True, tmp_path)
    env=DynamicDeliveryEnv(inst['output_directory'])
    env.reset()
    assert all(p.status in {'UNRELEASED','PENDING_ASSIGNMENT'} for p in env.parcels.values())
    unreleased=[p.parcel_id for p in env.parcels.values() if p.status=='UNRELEASED']
    queued=str(env.pending_truck_tasks)+str(env.bus_terminal_ready)+str(env.waiting_station_parcels)
    assert all(pid not in queued for pid in unreleased)
    assert env.current_decision.agent == 'assignment'
