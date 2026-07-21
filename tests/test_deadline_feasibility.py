from data_pipeline.build_instance import build_instance

def test_formal_parcels_have_reachability_and_nominal_mode(tmp_path):
    inst=build_instance('configs/paper/base_medium.yaml', True, tmp_path)
    import csv, pathlib
    rows=list(csv.DictReader(open(pathlib.Path(inst['output_directory'])/'parcels.csv')))
    assert rows
    assert all(r['reachable_station_ids'] for r in rows)
    assert all(r['nominal_feasible_modes'] for r in rows)
    assert {float(r['weight_kg']) for r in rows} <= {x/2 for x in range(1,10)}
