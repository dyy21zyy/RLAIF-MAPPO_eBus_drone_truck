from envs.dynamics.passenger_dynamics import PassengerBusManifest, add_onboard_extra_delay

def test_extra_dwell_uses_onboard_at_start():
    bus=PassengerBusManifest(total_onboard_passengers=5)
    assert add_onboard_extra_delay(bus,4,category='charging') == 20
    bus.total_onboard_passengers += 3
    assert bus.onboard_additional_delay_passenger_minutes == 20
