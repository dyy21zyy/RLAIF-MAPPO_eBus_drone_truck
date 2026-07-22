import pytest
from envs.dynamics.passenger_dynamics import PassengerArrivalEvent, PassengerArrivalIndex, PassengerStopRuntimeState, PassengerSystemRuntime

def test_queue_integrates_once_and_not_backward():
    stops={'s':PassengerStopRuntimeState('s')}; idx=PassengerArrivalIndex([PassengerArrivalEvent('e','s','d',10,2)])
    rt=PassengerSystemRuntime(stops,idx)
    assert rt.integrate_all_queues_until(10) == 0
    assert rt.integrate_all_queues_until(15) == 10
    assert rt.integrate_all_queues_until(15) == 0
    with pytest.raises(ValueError): rt.integrate_all_queues_until(14)
