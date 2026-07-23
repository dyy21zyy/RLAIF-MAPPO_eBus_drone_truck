import pytest
from evaluation.preformal_part3_gates import REQUIRED_METRICS, BenchmarkIntegrityError, validate_benchmark_row

def metrics(v=1):
    return {k:{'value':v,'available':True,'finite':True,'source':'runtime','formula':'measured','legitimate_zero':v==0} for k in REQUIRED_METRICS}

def test_successful_row_requires_env_step_and_positive_transition_count():
    with pytest.raises(BenchmarkIntegrityError): validate_benchmark_row({'status':'success','transition_count':0,'formal_metrics':metrics()})

def test_missing_metric_fails_and_legitimate_zero_passes_non_placeholder():
    m=metrics(); del m['released_parcels']
    with pytest.raises(BenchmarkIntegrityError): validate_benchmark_row({'status':'success','transition_count':1,'formal_metrics':m})
    z=metrics(0); z['released_parcels']['value']=1
    validate_benchmark_row({'status':'success','transition_count':1,'formal_metrics':z})

def test_fixed_zero_placeholder_fails():
    with pytest.raises(BenchmarkIntegrityError): validate_benchmark_row({'status':'success','transition_count':1,'formal_metrics':metrics(0)})
