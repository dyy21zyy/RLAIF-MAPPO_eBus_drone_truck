from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEMPLATE = Path('configs/diagnostic/preformal_gate.template.yaml')

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--output', default='results/diagnostic/preformal_gate/config.yaml'); p.add_argument('--force', action='store_true')
    a=p.parse_args(argv); out=Path(a.output)
    if out.exists() and not a.force: raise SystemExit(f'{out} exists; pass --force')
    out.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(TEMPLATE, out); print(out); return 0
if __name__=='__main__': raise SystemExit(main())
