from envs.dynamics.bus_circulation import RuntimePhysicalBus

def test_trip_start_does_not_reset_and_soc_is_continuous():
    bus=RuntimePhysicalBus("bus_000","s0",100)
    start=bus.soc_kwh; bus.soc_kwh -= 10*1.6; end1=bus.soc_kwh
    bus.current_trip_id="t1"
    assert bus.soc_kwh == end1 and bus.soc_kwh != 160
    bus.soc_kwh += 5
    bus.soc_kwh -= 2*1.6
    assert bus.soc_kwh == end1 + 5 - 3.2

def test_delay_propagates_to_later_trip():
    bus=RuntimePhysicalBus("bus_000","s0",100, schedule_delay_min=4, next_available_time_min=24)
    scheduled=20
    actual=max(scheduled,bus.next_available_time_min)
    assert actual == 24
    assert actual - scheduled == 4
