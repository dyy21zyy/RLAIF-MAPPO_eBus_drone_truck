from evaluation.experiment_aggregation import paired_differences

def test_paired_differences_are_calculated_before_averaging():
    rows=[{'status':'evaluation_success','variant_id':'b','scenario_family_id':'s1','master_seed':1,'environment_reward':2},{'status':'evaluation_success','variant_id':'c','scenario_family_id':'s1','master_seed':1,'environment_reward':5}]
    out=paired_differences(rows, baseline_selector=lambda r:r['variant_id']=='b', comparison_selector=lambda r:r['variant_id']=='c')
    assert out['pairs'][0]['difference']==3
def test_insufficient_sample_status_is_reported():
    rows=[{'status':'evaluation_success','variant_id':'b','scenario_family_id':'s1','master_seed':1,'environment_reward':2},{'status':'evaluation_success','variant_id':'c','scenario_family_id':'s1','master_seed':1,'environment_reward':5}]
    assert paired_differences(rows, baseline_selector=lambda r:r['variant_id']=='b', comparison_selector=lambda r:r['variant_id']=='c')['summary']['status']=='insufficient_samples'
