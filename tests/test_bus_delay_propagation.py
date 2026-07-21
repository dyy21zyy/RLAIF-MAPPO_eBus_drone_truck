from types import SimpleNamespace as N
from envs.dynamics.bus_operations import passenger_delay_cost

def test_loading_and_charging_delay_can_propagate_to_physical_bus_later_trip():
    bus=N(schedule_delay_min=0,next_available_time_min=0)
    bus.schedule_delay_min += 1.5
    bus.next_available_time_min = 10 + bus.schedule_delay_min
    assert bus.next_available_time_min == 11.5
    assert passenger_delay_cost(bus.schedule_delay_min, 3, 0) == 4.5
