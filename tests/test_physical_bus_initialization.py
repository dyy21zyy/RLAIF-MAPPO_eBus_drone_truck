from envs.dynamics.bus_circulation import sample_initial_energy

def test_initial_soc_range_reproducible_and_seeded():
    ids=["bus_001","bus_000"]
    a=sample_initial_energy(ids,123); b=sample_initial_energy(ids,123); c=sample_initial_energy(ids,124)
    assert a == b
    assert a != c
    assert all(88 <= v <= 136 for v in a.values())
