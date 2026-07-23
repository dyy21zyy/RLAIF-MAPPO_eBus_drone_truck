from evaluation.preformal_part3_gates import paired_statistics

def rr(mid,s,v):
    return {'method_id':mid,'status':'success','scenario_id':s,'scenario_content_hash':s,'instance_hash':s,'scenario_manifest_hash':'m','scenario_bank_hash':'b','artifact_hashes':{},'formal_metrics':{'environment_reward':{'value':v}}}

def test_paired_differences_before_averaging_and_insufficient_samples():
    out=paired_statistics([rr('a','s1',1),rr('b','s1',3),rr('a','s2',2),rr('b','s2',5)],lambda r:r['method_id']=='a',lambda r:r['method_id']=='b')
    assert [p['paired_difference'] for p in out['metrics']['environment_reward']['pairs']]==[2,3]
    assert out['metrics']['environment_reward']['summary']['mean_paired_difference']==2.5
    assert paired_statistics([rr('a','s1',1),rr('b','s1',2)],lambda r:r['method_id']=='a',lambda r:r['method_id']=='b')['metrics']['environment_reward']['summary']['status']=='insufficient_samples'
