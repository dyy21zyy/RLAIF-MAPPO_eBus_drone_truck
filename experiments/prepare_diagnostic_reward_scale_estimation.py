from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from experiments.build_scenario_bank import build_bank

CONFIG = Path(__file__).parents[1] / "configs/shanghai_small.yaml"

def prepare(output_root: str|Path, *, force: bool=False):
    out=Path(output_root)
    if out.exists() and force: shutil.rmtree(out)
    if out.exists() and any(out.iterdir()) and not force: raise FileExistsError(f"{out} exists; use --force")
    scen=out/"scenarios"/"train"
    manifest=build_bank(CONFIG, "train", 1, 9100, scen, fallback=True, run_classification="diagnostic", force=True)
    cfg=Path("configs/diagnostic/reward_scale_estimation.template.yaml").read_text()
    cfg=cfg.replace("__GENERATED_TRAIN_BANK_MANIFEST__", str(scen/"scenario_bank_manifest.json")).replace("__GENERATED_SCALE_ARTIFACT__", str(out/"reward_reference_scales.json"))
    resolved=out/"reward_scale_estimation.resolved.yaml"; resolved.parent.mkdir(parents=True, exist_ok=True); resolved.write_text(cfg)
    shutil.rmtree(scen/"_generated", ignore_errors=True)
    return {"scenario_bank": str(scen/"scenario_bank_manifest.json"), "config": str(resolved), "output": str(out/"reward_reference_scales.json"), "bank_hash": manifest["bank_hash"]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-root", default="results/diagnostic/reward_scales"); ap.add_argument("--force", action="store_true")
    args=ap.parse_args(); print(json.dumps(prepare(args.output_root, force=args.force), indent=2))
if __name__=="__main__": main()
