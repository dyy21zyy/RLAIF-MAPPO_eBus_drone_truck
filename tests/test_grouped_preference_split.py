from rlaif.grouped_split import grouped_split

def test_same_state_pairs_never_cross_splits():
    rows=[{"scenario_id":"x","episode_id":"1","state_id":"s","i":i} for i in range(3)]+[{"scenario_id":"x","episode_id":"1","state_id":f"s{i}","i":i} for i in range(3,8)]
    split=grouped_split(rows,seed=3)
    seen={}
    for name,rs in split["records"].items():
        for r in rs:
            key=(r['scenario_id'],r['episode_id'],r['state_id'])
            assert key not in seen or seen[key]==name
            seen[key]=name
    assert split['hash']
