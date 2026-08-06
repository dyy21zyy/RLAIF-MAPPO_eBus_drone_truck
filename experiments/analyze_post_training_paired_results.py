import argparse,json
from pathlib import Path
from .post_training.analysis import analyze
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("input",type=Path); p.add_argument("output",type=Path); p.add_argument("--metrics",nargs="+",required=True); a=p.parse_args(argv)
 rows=[json.loads(x) for x in a.input.read_text().splitlines() if x.strip()]; result=analyze(rows,a.metrics); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+"\n")
if __name__=="__main__":main()
