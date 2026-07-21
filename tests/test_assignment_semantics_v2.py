from data_pipeline.build_instance import build_instance
from envs.delivery_env import DynamicDeliveryEnv
from envs.state_builder import build_assignment_decision_surface
import copy

def test_assignment_surface_and_tbd_task_do_not_bind_trip_or_vehicle(tmp_path):
    inst=build_instance('configs/shanghai_small.yaml', True, tmp_path)
    env=DynamicDeliveryEnv(inst['output_directory']); obs,_=env.reset()
    parcel=env.parcels[obs['entity_id']]
    surface=build_assignment_decision_surface(env, parcel, obs['action_mask'])
    assert all(c.description == 'TD' or c.description.startswith(('TBD_','TLD_')) for c in surface.candidates)
    action=next((i for i,c in enumerate(surface.candidates) if c.description.startswith('TBD_') and c.feasible), None)
    if action is None: return
    env.step(action)
    task=env.pending_truck_tasks[-1]
    assert 'trip_id' not in task and 'bus_id' not in task and 'truck_id' not in task
    assert task['kind']=='bus_terminal_feeder' and task['terminal_transfer_required'] is True and task['station_id']

def test_assignment_mask_side_effect_free(tmp_path):
    inst=build_instance('configs/shanghai_small.yaml', True, tmp_path)
    env=DynamicDeliveryEnv(inst['output_directory']); obs,_=env.reset()
    before=(copy.deepcopy(env.bus_freight_kg), copy.deepcopy(env.pending_bus_parcels), copy.deepcopy(env.pending_truck_tasks))
    env._assignment_mask(env.parcels[obs['entity_id']])
    after=(env.bus_freight_kg, env.pending_bus_parcels, env.pending_truck_tasks)
    assert before == after
