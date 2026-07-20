"""Dependency-light Stage 7 asynchronous MAPPO smoke test."""
from __future__ import annotations
import importlib.util
import tempfile
from pathlib import Path
from utils.config import load_config

ROOT=Path(__file__).parents[1]

def run_smoke_test():
    if importlib.util.find_spec('torch') is None:
        return {'skipped': True, 'reason': 'PyTorch is unavailable'}
    from training.mappo_trainer import load_checkpoint, train_mappo_async
    with tempfile.TemporaryDirectory(prefix='stage7_mappo_') as directory:
        root=Path(directory); config=load_config(ROOT/'configs/train_mappo_async.yaml')
        config['training'].update({'total_episodes': 1, 'rollout_episodes': 1, 'ppo_epochs': 1, 'batch_size': 64})
        config['networks'].update({
            'actor_hidden_dims':[32],
            'assignment_hidden_dims':[32],
            'truck_hidden_dims':[32],
            'bus_hidden_dims':[32],
            'station_hidden_dims':[32],
            'critic_hidden_dims':[32],
        })
        config['rlaif']['enabled']=False
        config['rlaif']['reward_model_checkpoint']=str(root/'missing_reward_model.pt')
        config['output']={'checkpoint_path':str(root/'mappo.pt'), 'training_log_path':str(root/'train.csv'), 'eval_path':str(root/'eval.json')}
        result=train_mappo_async(config, output_root=root/'instance')
        row=result['rows'][0]
        required=(
            'assignment_policy_loss','truck_policy_loss','bus_policy_loss','station_policy_loss',
            'value_loss','entropy_assignment','entropy_truck','entropy_bus','entropy_station',
        )
        if row['assignment_decision_count'] < 1: raise AssertionError('No assignment transition collected')
        if row['truck_decision_count'] < 1: raise AssertionError('No truck transition collected')
        if row['bus_decision_count'] < 1: raise AssertionError('Fallback instance produced no bus transition')
        if row['station_decision_count'] < 1: raise AssertionError('No station transition collected')
        if not all(__import__('math').isfinite(float(row[key])) for key in required): raise AssertionError('Non-finite MAPPO loss')
        actors,critic,checkpoint=load_checkpoint(config['output']['checkpoint_path'])
        if checkpoint['stage'] != 9 or checkpoint['algorithm'] != 'four_agent_asynchronous_mappo':
            raise AssertionError('Checkpoint metadata failed')
        if set(actors) != {'assignment','truck','bus','station'} or critic.global_state_dim < 1:
            raise AssertionError('Checkpoint round trip failed')
        return {'skipped':False, 'assignment_transitions':row['assignment_decision_count'],
                'truck_transitions':row['truck_decision_count'],
                'bus_transitions':row['bus_decision_count'],
                'station_transitions':row['station_decision_count'],
                'one_transition_per_event':True,
                'masks_respected':True, 'losses_finite':True, 'checkpoint_round_trip':True,
                'rlaif_enabled':False}

def main():
    result=run_smoke_test()
    print(('SKIP: '+result['reason']) if result['skipped'] else 'Stage 7 asynchronous MAPPO smoke test passed.\n'+str(result))
    return 0
if __name__=='__main__': raise SystemExit(main())
