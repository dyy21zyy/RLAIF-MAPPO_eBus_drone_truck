import pytest
from evaluation.preformal_part3_gates import PreformalConfigDifferenceError, validate_config_differences

def test_only_intended_config_field_differs():
    assert validate_config_differences({'a':1,'lr':2},{'a':3,'lr':2},{'a'})['unexpected_config_differences']==[]
    with pytest.raises(PreformalConfigDifferenceError): validate_config_differences({'a':1,'lr':2},{'a':3,'lr':4},{'a'})
