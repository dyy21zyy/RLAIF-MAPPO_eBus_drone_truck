from envs.dynamics.passenger_dynamics import PassengerArrivalEvent, PassengerArrivalIndex, PassengerBusManifest, PassengerStopRuntimeState, process_bus_stop

def test_arrivals_during_dwell_board_extend_and_terminate():
    events=[PassengerArrivalEvent('e1','s0','s1',0.01,1), PassengerArrivalEvent('e2','s0','s1',0.07,1)]
    stop=PassengerStopRuntimeState('s0', {'s1':1}, 1)
    bus=PassengerBusManifest(passenger_capacity=80)
    res=process_bus_stop(stop,bus,PassengerArrivalIndex(events),0.0, boarding_time_sec=3.0)
    assert res.boarding_count == 3
    assert res.realized_dwell_min >= 0.15
    assert not res.terminated_by_iteration_cap

def test_iteration_cap_protects_infinite_loops():
    events=[PassengerArrivalEvent(f'e{i}','s0','s1',0.04*i,1) for i in range(2000)]
    stop=PassengerStopRuntimeState('s0')
    bus=PassengerBusManifest(passenger_capacity=5000)
    res=process_bus_stop(stop,bus,PassengerArrivalIndex(events),0.0,max_iterations=1)
    assert res.terminated_by_iteration_cap
