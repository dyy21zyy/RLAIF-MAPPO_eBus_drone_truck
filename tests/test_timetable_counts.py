from pathlib import Path
from data_pipeline.build_instance import build_instance

def _counts(cfg, tmp_path):
    inst=build_instance(cfg, fallback=True, output_root=tmp_path)
    trips=inst['counts']['bus_trips']; freight=sum(1 for r in __import__('csv').DictReader(open(Path(inst['output_directory'])/'bus_trips.csv')) if r['freight_allowed']=='True')
    return trips, freight, max(float(r['start_time']) for r in __import__('csv').DictReader(open(Path(inst['output_directory'])/'bus_trips.csv')))

def test_medium_timetable_counts(tmp_path):
    assert _counts('configs/paper/base_medium.yaml', tmp_path) == (36,12,350.0)

def test_large_timetable_counts(tmp_path):
    trips, freight, last = _counts('configs/paper/base_large.yaml', tmp_path)
    assert (trips, freight) == (45,16)
    assert last < 360
