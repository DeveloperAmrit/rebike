"""Pick the final agent among saved checkpoints using validation days.

    python scripts/select_checkpoint.py --name ppo_v2      # -> models/ppo_v2_selected.zip

Every candidate (models/checkpoints/<name>_*_steps.zip, models/<name>.zip and
models/<name>_best.zip) is scored deterministically on the SAME `--episodes`
validation days drawn from the *training* demand model (seeds from ppo.val_seed).
The held-out test month is never touched. Writes results/<name>_checkpoints.csv.
"""
import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from rebike.config import load_config
from rebike.data import DemandModel
from rebike.policies import PPOPolicy, env_for_model, run_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--name", default="ppo_v2")
    p.add_argument("--episodes", type=int, default=30)
    args = p.parse_args()
    cfg = load_config(args.config)
    pc = cfg["ppo"]
    mdir, out = Path(pc["model_dir"]), Path(cfg["eval"]["out_dir"])
    demand = DemandModel.load(Path(cfg["env"].model_path))

    cands = sorted(mdir.glob(f"checkpoints/{args.name}_*_steps.zip"),
                   key=lambda q: int(re.findall(r"_(\d+)_steps", q.name)[0]))
    cands += [q for q in (mdir / f"{args.name}.zip", mdir / f"{args.name}_best.zip") if q.exists()]
    seeds = [pc.get("val_seed", 7000) + 100 + i for i in range(args.episodes)]
    rows = []
    for c in cands:
        env = env_for_model(c, cfg["env"], demand)
        pol = PPOPolicy(c, env)
        res = [run_episode(env, pol, s) for s in seeds]
        rows.append(dict(candidate=c.name, val_return=np.mean([r["ret"] for r in res]),
                         val_service=np.mean([r["service_rate"] for r in res])))
        print(f"[select] {c.name:40s} return={rows[-1]['val_return']:8.1f} service={rows[-1]['val_service']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{args.name}_checkpoints.csv", index=False)
    best = df.loc[df.val_return.idxmax(), "candidate"]
    src = next(c for c in cands if c.name == best)
    shutil.copy(src, mdir / f"{args.name}_selected.zip")
    print(f"[select] selected {best} -> {mdir / (args.name + '_selected.zip')}")


if __name__ == "__main__":
    main()
