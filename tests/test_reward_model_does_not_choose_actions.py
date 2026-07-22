from evaluation.formal_policy_registry import get_formal_policy_spec, validate_unique_learned_checkpoints, PolicyCheckpointValidationError
from evaluation.runner import build_reward_registry_for_method, select_action_with_policy, score_rlaif_decomposition
from rlaif.reward_registry import empty_rlaif_training_totals
import pytest
class P:
    def act(self, obs, mask, deterministic=True): return (1, None)
def test_reward_models_do_not_select_actions():
    assert select_action_with_policy(P(), {'x':1}, [True, True]) == 1
    assert select_action_with_policy(P(), {'x':1}, [True, True]) == 1
def test_empty_rlaif_decomposition_has_agents():
    t=empty_rlaif_training_totals()
    assert all(f'rlaif_{a}_weighted' in t for a in ('assignment','truck','bus','station'))
    assert t['rlaif_total_weighted']==0
