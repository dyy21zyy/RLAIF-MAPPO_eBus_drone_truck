from envs.dynamics.passenger_dynamics import PassengerBusManifest, PassengerStopRuntimeState

def test_boarding_never_exceeds_capacity_80():
    stop=PassengerStopRuntimeState('s0', {'s1':100}, 100)
    bus=PassengerBusManifest(passenger_capacity=80)
    assert bus.board_from_stop(stop) == 80
    assert bus.total_onboard_passengers == 80
    assert stop.total_waiting == 20
