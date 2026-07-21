from envs.dynamics.bus_operations import passenger_delay_cost

def test_more_passengers_more_delay_cost_same_charge():
    assert passenger_delay_cost(2,onboard=10,waiting=5) > passenger_delay_cost(2,onboard=1,waiting=1)
