import pytest
from experiments.run_paper_ablation import validate_ablations
def test_ablation_validation():
    validate_ablations({'ablations':[{'name':'no_rlaif','config_difference':{'x':1},'requires_retraining':True,'checkpoint':'a','checkpoint_hash':'ha'}]})
    with pytest.raises(ValueError): validate_ablations({'ablations':[{'name':'no_rlaif','config_difference':{},'requires_retraining':True,'checkpoint':'a','checkpoint_hash':'h'},{'name':'full_rlaif','config_difference':{'x':2},'requires_retraining':True,'checkpoint':'b','checkpoint_hash':'h'}]})
