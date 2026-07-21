import pytest
from utils.config import load_config
from training.config_resolver import TrainingConfigError, resolve_mappo_training_config

def test_conflicting_entropy_alias_fails():
    c=load_config('configs/paper/train_mappo_env.yaml'); c['training']['ent_coef']=9
    with pytest.raises(TrainingConfigError): resolve_mappo_training_config(c)

def test_conflicting_value_alias_fails():
    c=load_config('configs/paper/train_mappo_env.yaml'); c['training']['vf_coef']=9
    with pytest.raises(TrainingConfigError): resolve_mappo_training_config(c)

def test_rollout_gt_total_fails():
    c=load_config('configs/paper/train_mappo_env.yaml'); c['training']['rollout_episodes']=99; c['training']['total_episodes']=1
    with pytest.raises(TrainingConfigError): resolve_mappo_training_config(c)
