"""Post-training benchmark CLI; dry unless --execute is supplied."""
import argparse
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--execute",action="store_true"); a=p.parse_args(argv)
 if not a.execute: print("DRY RUN: planned 900 rows; no episodes executed"); return
 raise SystemExit("Supply validated registry/scenario artifacts through the formal orchestrator")
if __name__=="__main__":main()
