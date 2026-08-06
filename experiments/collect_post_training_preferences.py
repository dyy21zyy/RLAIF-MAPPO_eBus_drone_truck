"""Validate collected assignment candidate JSONL (no evaluator calls)."""
import argparse,json
from pathlib import Path
from .post_training.preference_collection import validate_records
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("input",type=Path); a=p.parse_args(argv)
 rows=[json.loads(x) for x in a.input.read_text().splitlines() if x.strip()]; validate_records(rows); print(f"validated {len(rows)} assignment-only base-policy pairs")
if __name__=="__main__": main()
