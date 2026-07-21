from envs.dynamics.passenger_dynamics import PassengerArrivalIndex, PassengerBusManifest, PassengerStopRuntimeState, process_bus_stop

def test_waiting_passenger_minutes_integrated_correctly():
    stop=PassengerStopRuntimeState('s0', {'s1':4}, 4)
    stop.integrate_waiting_until(10.0)
    assert stop.cumulative_waiting_passenger_minutes == 40.0

def test_same_extra_dwell_higher_onboard_cost_with_more_passengers():
    a=PassengerBusManifest(total_onboard_passengers=10)
    b=PassengerBusManifest(total_onboard_passengers=20)
    a.onboard_additional_delay_passenger_minutes += a.total_onboard_passengers*2
    b.onboard_additional_delay_passenger_minutes += b.total_onboard_passengers*2
    assert b.onboard_additional_delay_passenger_minutes == 2*a.onboard_additional_delay_passenger_minutes

def test_normal_scheduled_travel_is_not_passenger_delay():
    bus=PassengerBusManifest(total_onboard_passengers=30)
    assert bus.onboard_additional_delay_passenger_minutes == 0.0

def test_passenger_dwell_propagates_and_state_persists():
    stop=PassengerStopRuntimeState('s0', {'s1':10}, 10)
    bus=PassengerBusManifest(passenger_capacity=80)
    res=process_bus_stop(stop,bus,PassengerArrivalIndex([]),5.0)
    downstream_arrival=10.0 + res.realized_dwell_min
    assert downstream_arrival > 10.0
    assert bus.total_onboard_passengers == 10
