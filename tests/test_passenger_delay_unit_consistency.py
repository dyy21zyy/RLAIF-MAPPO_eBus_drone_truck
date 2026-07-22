from envs.dynamics.passenger_dynamics import PassengerArrivalEvent, PassengerArrivalIndex, PassengerStopRuntimeState, PassengerBusManifest, process_bus_stop

def test_normal_dwell_not_onboard_extra_delay():
    stop=PassengerStopRuntimeState('s'); bus=PassengerBusManifest(passenger_capacity=10)
    idx=PassengerArrivalIndex([PassengerArrivalEvent('e','s','d',0,2)])
    res=process_bus_stop(stop,bus,idx,0,boarding_time_sec=3,alighting_time_sec=1.5)
    assert res.realized_dwell_min == 0.1
    assert bus.onboard_additional_delay_passenger_minutes == 0
