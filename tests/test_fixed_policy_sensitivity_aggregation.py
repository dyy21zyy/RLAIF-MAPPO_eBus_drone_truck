import pytest
from experiments.aggregate_fixed_policy_sensitivity import bootstrap_ci, holm_adjust, paired_test, rank_biserial, validate_rows

def make_rows(seeds=(1,2,3),indices=(0,1),values=(1.0,1.5),classification="formal",mode="fixed_policy_robustness"):
    rows=[]
    for value in values:
      for seed in seeds:
       for i in indices:
        rows.append({"sensitivity_family":"passenger","parameter_value":str(value),"policy_seed":str(seed),"paired_scenario_index":str(i),"scenario_id":f"{value}-{i}","scenario_content_hash":f"h-{value}-{i}","sensitivity_mode":mode,"run_classification":classification,"status":"success","fallback_count":"0","policy_checkpoint_hash":f"p{seed}","scenario_bank_hash":f"b{value}"})
    return rows

def test_validation_rejects_missing_policy_seed():
    with pytest.raises(ValueError,match="policy seeds"): validate_rows(make_rows(seeds=(1,2)),expected_scenarios=2)

def test_validation_rejects_missing_paired_scenario():
    with pytest.raises(ValueError,match="paired scenarios"): validate_rows(make_rows(indices=(0,)),expected_scenarios=2)

def test_validation_rejects_duplicate_and_mixed_modes():
    rows=make_rows();
    with pytest.raises(ValueError,match="duplicate"): validate_rows(rows+[rows[0]],expected_scenarios=2)
    rows[0]["sensitivity_mode"]="retrained_policy_sensitivity"
    with pytest.raises(ValueError,match="mixing"): validate_rows(rows,expected_scenarios=2)

def test_validation_rejects_nonzero_fallback_and_diagnostics():
    rows=make_rows(); rows[0]["fallback_count"]="1"
    with pytest.raises(ValueError,match="fallback"): validate_rows(rows,expected_scenarios=2)
    rows=make_rows(classification="diagnostic")
    with pytest.raises(ValueError,match="diagnostic"): validate_rows(rows,expected_scenarios=2)

def test_bootstrap_is_reproducible():
    assert bootstrap_ci([1,2,3,4],seed=9,resamples=500)==bootstrap_ci([1,2,3,4],seed=9,resamples=500)

def test_rank_biserial_and_all_zero_handling():
    assert rank_biserial([1,2,-1])==pytest.approx(.5)
    assert paired_test([1,1],[1,1])=={"p_value":1.0,"rank_biserial":0.0,"all_zero_differences":True}

def test_holm_adjustment_is_monotone_in_sorted_p_order():
    adjusted=holm_adjust([.01,.04,.03]); assert adjusted==pytest.approx([.03,.06,.06])
