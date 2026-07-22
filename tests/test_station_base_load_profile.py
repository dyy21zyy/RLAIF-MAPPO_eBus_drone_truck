from envs.dynamics.station_power import StationBaseLoadInterval, StationBaseLoadProfile

def test_profile_load_and_boundary():
    p=StationBaseLoadProfile([StationBaseLoadInterval('h','i0',0,15,100),StationBaseLoadInterval('h','i1',15,30,150)])
    assert p.load_at('h',10)==100 and p.load_at('h',16)==150
    assert p.next_boundary_after('h',10)==15
