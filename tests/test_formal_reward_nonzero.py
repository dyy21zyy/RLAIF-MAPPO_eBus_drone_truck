import math, pytest
from utils.config import load_config
from training.config_resolver import TrainingConfigError, validate_nonzero_formal_reward

def test_formal_reward_present_nonzero():
    validate_nonzero_formal_reward(load_config('configs/paper/base_medium.yaml'))

def test_missing_all_zero_negative_invalid():
    c={}
    with pytest.raises(TrainingConfigError): validate_nonzero_formal_reward(c)
    base=load_config('configs/paper/base_medium.yaml');
    for k in list(base['reward']):
        if isinstance(base['reward'][k], (int,float)): base['reward'][k]=0.0
    with pytest.raises(TrainingConfigError): validate_nonzero_formal_reward(base)
    base=load_config('configs/paper/base_medium.yaml'); base['reward']['truck_cost']=-1
    with pytest.raises(TrainingConfigError): validate_nonzero_formal_reward(base)
    base=load_config('configs/paper/base_medium.yaml'); base['reward']['truck_cost']=math.nan
    with pytest.raises(TrainingConfigError): validate_nonzero_formal_reward(base)
    base=load_config('configs/paper/base_medium.yaml'); base['reward']['truck_cost']=True
    with pytest.raises(TrainingConfigError): validate_nonzero_formal_reward(base)
