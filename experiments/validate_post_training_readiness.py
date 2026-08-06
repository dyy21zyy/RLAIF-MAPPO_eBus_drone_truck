import argparse,json
from pathlib import Path
from .post_training.readiness import validate_seed_pair
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("base",type=Path); p.add_argument("continued",type=Path); p.add_argument("rlaif",type=Path); a=p.parse_args(argv)
 validate_seed_pair(*[json.loads(x.read_text()) for x in (a.base,a.continued,a.rlaif)]); print("post-training readiness: PASS")
if __name__=="__main__":main()
