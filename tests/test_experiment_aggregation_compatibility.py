from evaluation.experiment_aggregation import aggregate_compatible

def test_diagnostic_and_formal_rows_are_separated():
    rows=[{'status':'evaluation_success','experiment_kind':'ablation','run_classification':'diagnostic','variant_id':'a','environment_reward':1},{'status':'evaluation_success','experiment_kind':'ablation','run_classification':'formal','variant_id':'a','environment_reward':2}]
    assert len(aggregate_compatible(rows))==2
def test_fixed_policy_and_retrained_sensitivity_are_separated():
    rows=[{'status':'evaluation_success','experiment_kind':'sensitivity','protocol':'fixed_policy_robustness','variant_id':'a','environment_reward':1},{'status':'evaluation_success','experiment_kind':'sensitivity','protocol':'retrained_policy_sensitivity','variant_id':'a','environment_reward':2}]
    assert len(aggregate_compatible(rows))==2
def test_assignment_and_all_agent_rlaif_remain_separated():
    rows=[{'status':'evaluation_success','experiment_kind':'ablation','variant_id':'x','policy_rlaif_scope':'assignment','environment_reward':1},{'status':'evaluation_success','experiment_kind':'ablation','variant_id':'x','policy_rlaif_scope':'all','environment_reward':2}]
    assert len(aggregate_compatible(rows))==2
def test_aggregation_excludes_failed_rows():
    assert aggregate_compatible([{'status':'evaluation_failed','variant_id':'a','environment_reward':1}])==[]
