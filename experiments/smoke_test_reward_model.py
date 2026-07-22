from __future__ import annotations
import json, shutil
from pathlib import Path
from experiments.generate_smoke_reward_preferences import generate_rows
from experiments.train_all_reward_models import main as train_all
from training.reward_model_wrapper import RewardCheckpointValidationError, load_strict_agent_reward_checkpoint

def main() -> int:
    out=Path('results/smoke/reward_models'); out.mkdir(parents=True,exist_ok=True)
    prefs=out/'preferences.jsonl'; prefs.write_text('\n'.join(json.dumps(r) for r in generate_rows(120))+'\n')
    rc=train_all(['--preferences',str(prefs),'--config-dir','configs/smoke','--output-dir',str(out)])
    ck=out/'reward_assignment.pt'; assert ck.exists()
    data,model=load_strict_agent_reward_checkpoint(ck,agent_type='assignment',formal=False)
    assert data['run_classification']=='smoke' and data['validation_status']=='smoke_only' and data.get('model_state_dict')
    try: load_strict_agent_reward_checkpoint(ck,agent_type='assignment',formal=True)
    except RewardCheckpointValidationError: pass
    else: raise AssertionError('formal loading accepted smoke checkpoint')
    print(json.dumps({'smoke_checkpoint':str(ck),'validation_accuracy':data['validation_metrics']['pairwise_accuracy'],'test_accuracy':data['test_metrics']['pairwise_accuracy'],'loss_before_after':'see training_history.csv','best_epoch':data['best_epoch']}))
    return 0
if __name__=='__main__': raise SystemExit(main())
# Legacy fixture helpers retained for existing Stage 5 tests.
try:
    from rlaif.preference_dataset import ACTION_FEATURE_KEYS, write_jsonl
    def _action(action_id: int, name: str, offset: float) -> dict[str, object]:
        values = {key: float(index + offset) for index, key in enumerate(ACTION_FEATURE_KEYS)}
        values['feasible_flag'] = 1.0
        return {'action_id': action_id, 'action_name': name, **values, 'infeasibility_reasons': []}
    def build_fixture(directory: Path, count: int = 10) -> tuple[Path, Path]:
        states, preferences = [], []
        for index in range(count):
            state_id = f'smoke-state-{index}'
            states.append({'state_id': state_id, 'feature_schema_version': 'v2', 'assignment_features': [index / 10, 0.1, 0.5, 1.0, 1.0, 0.0], 'candidate_action_features': {'TD': _action(0, 'TD', 0.2 + index / 100), 'TBD_station_01': _action(1, 'TBD_station_01', 1.2 + index / 100)}})
            preferences.append({'preference_id': f'smoke-pref-{index}', 'state_id': state_id, 'action_a': 'TD', 'action_b': 'TBD_station_01', 'chosen': 'TD', 'rejected': 'TBD_station_01', 'confidence': 0.9, 'validation_status': 'valid', 'usable_for_training': True, 'label_source': 'temporary_smoke_fixture'})
        states_path, preferences_path = directory / 'states.jsonl', directory / 'preferences.jsonl'
        write_jsonl(states_path, states); write_jsonl(preferences_path, preferences)
        return states_path, preferences_path
    def _config(directory: Path, states: Path, preferences: Path) -> dict[str, object]:
        return {'reward_model': {'action_emb_dim': 16, 'hidden_dims': [32, 16], 'dropout': 0.0}, 'training': {'batch_size': 8, 'epochs': 2, 'lr': 0.001, 'weight_decay': 0.0, 'seed': 42, 'early_stopping_patience': 2, 'min_delta': 0.0}, 'data': {'preferences_path': str(preferences), 'assignment_states_path': str(states), 'train_ratio': 0.8, 'val_ratio': 0.1, 'test_ratio': 0.1, 'min_confidence': 0.6, 'use_only_usable_for_training': True}, 'output': {'checkpoint_path': str(directory / 'reward_model.pt'), 'training_log_path': str(directory / 'training.csv'), 'eval_path': str(directory / 'eval.json')}}
except Exception:
    pass
