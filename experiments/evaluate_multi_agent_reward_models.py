from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); args=ap.parse_args(); print(json.dumps({'checkpoint':args.checkpoint,'exists':Path(args.checkpoint).exists()}))
if __name__=='__main__': main()
