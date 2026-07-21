from envs.dynamics.bus_circulation import RuntimePhysicalBus

def test_relocation_consumes_time_energy_and_layover_location_persists():
    bus=RuntimePhysicalBus("bus_000","terminal_a",100)
    end=30; relocation=5; layover=2; speed=30; e_per_km=1.6
    energy=relocation/60*speed*e_per_km
    bus.soc_kwh -= energy; bus.last_relocation_energy_kwh=energy; bus.current_location="terminal_b"; bus.next_available_time_min=end+relocation+layover
    assert bus.last_relocation_energy_kwh == 4
    assert bus.next_available_time_min == 37
    assert bus.current_location == "terminal_b"

def test_complete_depletion_detected():
    bus=RuntimePhysicalBus("bus_000","x",1)
    bus.soc_kwh -= 2
    bus.depleted = bus.soc_kwh <= 0
    assert bus.depleted
