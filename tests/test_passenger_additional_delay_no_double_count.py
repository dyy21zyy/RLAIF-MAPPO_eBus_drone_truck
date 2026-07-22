from envs.dynamics.passenger_dynamics import PassengerBusManifest, add_onboard_extra_delay

def test_components_sum_once():
    bus=PassengerBusManifest(total_onboard_passengers=2)
    loading=add_onboard_extra_delay(bus,1,category='loading')
    unloading=add_onboard_extra_delay(bus,2,category='unloading')
    charging=add_onboard_extra_delay(bus,3,category='charging')
    assert bus.onboard_additional_delay_passenger_minutes == loading+unloading+charging == 12
