import csv

import pytest
import yaml

from experiments.aggregate_fixed_policy_sensitivity import _write_csv, bootstrap_ci, holm_adjust, paired_test, rank_biserial, run, validate_rows

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

def test_write_csv_preserves_heterogeneous_metrics_and_deterministic_order(tmp_path):
    rows=[
        {"sensitivity_family":"passenger","parameter_value":1.0,"paired_scenario_index":0,"raw_successful_run_count":3,"fulfillment_rate":.9,"waiting_minutes_per_passenger":2.5,"onboard_delay_minutes_per_passenger":1.5},
        {"sensitivity_family":"parcel","parameter_value":60,"paired_scenario_index":0,"raw_successful_run_count":3,"fulfillment_rate":.8,"truck_distance":12.0,"drone_missions":4.0},
        {"sensitivity_family":"power","parameter_value":1100,"paired_scenario_index":0,"raw_successful_run_count":3,"station_peak_power":900.0,"peak_load_to_capacity_ratio":.82,"overload_kw_min":0.0},
    ]
    first=tmp_path/"first.csv"; second=tmp_path/"second.csv"
    _write_csv(first,rows); _write_csv(second,list(reversed(rows)))

    with first.open(newline="") as f: written=list(csv.DictReader(f)); first_header=written[0].keys()
    with second.open(newline="") as f: second_header=next(csv.reader(f))
    expected_metrics={"fulfillment_rate","waiting_minutes_per_passenger","onboard_delay_minutes_per_passenger","truck_distance","drone_missions","station_peak_power","peak_load_to_capacity_ratio","overload_kw_min"}
    assert list(first_header)[:4]==["sensitivity_family","parameter_value","paired_scenario_index","raw_successful_run_count"]
    assert expected_metrics <= set(first_header)
    assert list(first_header)==second_header
    assert written[0]["truck_distance"]==""
    assert written[1]["waiting_minutes_per_passenger"]==""
    assert written[2]["fulfillment_rate"]==""
    assert written[2]["overload_kw_min"]=="0.0"

def test_write_csv_handles_homogeneous_and_empty_rows(tmp_path):
    rows=[{"sensitivity_family":"passenger","metric":"fulfillment_rate","mean":.8},{"sensitivity_family":"passenger","metric":"waiting","mean":2.0}]
    path=tmp_path/"summary.csv"; _write_csv(path,rows)
    with path.open(newline="") as f: assert list(csv.DictReader(f))==[{key:str(value) for key,value in row.items()} for row in rows]
    empty=tmp_path/"empty.csv"; _write_csv(empty,[])
    assert empty.read_text()=="\n"

def test_run_writes_combined_paired_rows_for_all_families(tmp_path):
    config=yaml.safe_load(open("configs/paper/fixed_policy_sensitivity.yaml"))
    config["paired_scenarios"]["count"]=2; config["aggregation"]["bootstrap_resamples"]=20
    for spec in config["families"].values(): spec["values"]=[spec["baseline_value"],spec["baseline_value"]*1.5]
    config_path=tmp_path/"config.yaml"; config_path.write_text(yaml.safe_dump(config))
    evaluation=tmp_path/"output"/"evaluation"; evaluation.mkdir(parents=True)
    rows=[]
    for family,spec in config["families"].items():
      for value in spec["values"]:
       for seed in (1,2,3):
        for index in (0,1):
         row={"sensitivity_family":family,"parameter_value":value,"policy_seed":seed,"paired_scenario_index":index,"scenario_id":f"{family}-{value}-{index}","scenario_content_hash":f"h-{family}-{value}-{index}","sensitivity_mode":"fixed_policy_robustness","run_classification":"formal","status":"success","fallback_count":0,"policy_checkpoint_hash":f"p{seed}","scenario_bank_hash":f"b-{family}-{value}"}
         row.update({metric:seed+index for metric in spec["primary_metrics"]}); rows.append(row)
    with (evaluation/"episodes.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(row.keys() for row in rows)))); writer.writeheader(); writer.writerows(rows)

    run(config_path,tmp_path/"output")

    with (tmp_path/"output"/"summary"/"paired_scenario_means.csv").open(newline="") as f: paired=list(csv.DictReader(f))
    assert len(paired)==12
    assert set().union(*(spec["primary_metrics"] for spec in config["families"].values())) <= set(paired[0])
    for family in config["families"]:
        assert (tmp_path/"output"/"summary"/f"{family}_summary.csv").is_file()
        assert (tmp_path/"output"/"summary"/f"{family}_paired_tests.csv").is_file()
