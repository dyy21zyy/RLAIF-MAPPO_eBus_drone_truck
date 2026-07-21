from envs.dynamics.passenger_dynamics import PassengerBusManifest, PassengerStopRuntimeState

def test_board_alight_and_nonnegative_manifest():
    stop=PassengerStopRuntimeState('s0', {'s1':5,'s2':3}, 8)
    bus=PassengerBusManifest(passenger_capacity=80)
    assert bus.board_from_stop(stop) == 8
    assert bus.total_onboard_passengers == 8 and stop.total_waiting == 0
    assert bus.alight('s1') == 5
    assert bus.total_onboard_passengers == 3
    assert bus.alight('s1') == 0
    assert bus.total_onboard_passengers >= 0
