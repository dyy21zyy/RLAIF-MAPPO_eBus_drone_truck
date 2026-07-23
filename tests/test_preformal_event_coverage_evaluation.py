import pytest
from evaluation.preformal_part3_gates import BenchmarkIntegrityError, CANONICAL_EVENTS, validate_event_coverage

def test_canonical_events_counted_and_automatic_excluded():
    rep=validate_event_coverage([{'method_id':'m','scenario_id':'s','event_counts':{e:1 for e in CANONICAL_EVENTS}|{'AUTO':99},'decision_count_by_agent':{'bus':2}}])
    assert 'AUTO' not in rep['event_counts'] and rep['event_counts']['BUS_TERMINAL_DEPARTURE']==1 and rep['event_counts']['BUS_STATION_ARRIVAL']==1
    with pytest.raises(BenchmarkIntegrityError): validate_event_coverage([{'event_counts':{}}])
