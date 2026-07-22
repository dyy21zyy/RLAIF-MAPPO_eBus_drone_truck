import pytest, torch
from rlaif.reward_model_normalization import fit_feature_normalization, apply_feature_normalization

def test_means_fit_training_only_and_outliers_do_not_alter():
    train=torch.tensor([[1.,2.],[3.,2.]]); val=torch.tensor([[999.,999.]]); test=torch.tensor([[-999.,-999.]])
    n=fit_feature_normalization(train, feature_names=('a','b'))
    assert n.mean==(2.0,2.0); assert n.std[1]==1.0
    assert apply_feature_normalization(val,n,feature_names=('a','b')).shape==(1,2)
    assert apply_feature_normalization(test,n,feature_names=('a','b')).shape==(1,2)
def test_feature_name_mismatch_blocks_normalization():
    n=fit_feature_normalization(torch.tensor([[1.,2.]]), feature_names=('a','b'))
    with pytest.raises(ValueError): apply_feature_normalization(torch.tensor([[1.,2.]]), n, feature_names=('b','a'))
