from envs.dynamics.passenger_dynamics import PassengerArrivalEvent, PassengerArrivalIndex, PassengerStopRuntimeState, PassengerSystemRuntime

def test_unvisited_stop_accumulates_to_horizon():
    stops={'s':PassengerStopRuntimeState('s')}; rt=PassengerSystemRuntime(stops, PassengerArrivalIndex([PassengerArrivalEvent('e','s','d',100,3)]))
    rt.integrate_all_queues_until(480)
    assert stops['s'].cumulative_waiting_passenger_minutes == 3*(480-100)
