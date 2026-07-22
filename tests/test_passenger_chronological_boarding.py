import pytest
from envs.dynamics.passenger_dynamics import PassengerArrivalEvent, PassengerArrivalIndex, PassengerStopRuntimeState, PassengerBusManifest, process_bus_stop

def test_arrivals_during_dwell_board_and_cap_raises():
    idx=PassengerArrivalIndex([PassengerArrivalEvent('a','s','d',0,1),PassengerArrivalEvent('b','s','d',0.04,1)])
    bus=PassengerBusManifest(passenger_capacity=10); stop=PassengerStopRuntimeState('s')
    res=process_bus_stop(stop,bus,idx,0,boarding_time_sec=3,max_iterations=10)
    assert res.boarding_count == 2 and res.departure_time_min >= 0.1
    idx2=PassengerArrivalIndex([PassengerArrivalEvent(str(i),'x','d',i*0.01,1) for i in range(10)])
    capped = process_bus_stop(PassengerStopRuntimeState('x'), PassengerBusManifest(passenger_capacity=100), idx2, 0, boarding_time_sec=60, max_iterations=1)
    assert capped.terminated_by_iteration_cap
