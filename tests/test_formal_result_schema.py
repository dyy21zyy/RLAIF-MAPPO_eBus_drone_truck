from evaluation.formal_metric_validation import *
import pytest, math
def full():
    d={k:{'value':1.0,'availability':'available','source':k,'legitimate_zero':False} for k in REQUIRED_FORMAL_METRICS+RLAIF_FIELDS}
    for k in RLAIF_FIELDS: d[k]={'value':0.0,'availability':'available','source':k,'legitimate_zero':True}
    d['rlaif_total_weighted']={'value':0.0,'availability':'available','source':'sum','legitimate_zero':True}; d['combined_reward_total']={'value':1.0,'availability':'available','source':'sum','legitimate_zero':False}; d['environment_reward']={'value':1.0,'availability':'available','source':'env','legitimate_zero':False}
    return d
def test_missing_fails_and_zero_ok():
    d=full(); d.pop('runtime')
    with pytest.raises(FormalMetricValidationError): validate_formal_metrics(d)
    d=full(); d['overload_kw_min']={'value':0.0,'availability':'available','source':'counter','legitimate_zero':True}; d['battery_safety_violations']={'value':0,'availability':'available','source':'counter','legitimate_zero':True}; assert validate_formal_metrics(d)['metric_source_map']
def test_nonfinite_fails():
    d=full(); d['runtime']={'value':math.inf,'availability':'available','source':'timer'}
    with pytest.raises(FormalMetricValidationError): validate_formal_metrics(d)
