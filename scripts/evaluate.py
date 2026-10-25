"""Compare trained PPO agents against the baselines on held-out (test-month) demand.

    python scripts/evaluate.py --model ppo_v2=models/ppo_v2_best.zip
    python scripts/evaluate.py --model ppo_v2=models/ppo_v2_best.zip \
                               --model ppo_v1=models/ppo_rebike_best.zip \
                               --model ppo_nomask=models/ppo_nomask_best.zip

Every policy is run on the same seeded days (same demand, same morning state).
Policies: no rebalancing, threshold with default settings, threshold with tuned
settings (results/threshold_tuned.json), and every `--model NAME=PATH`.
Writes results/eval_summary.csv, results/eval_episodes.csv, results/eval_comparison.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from rebike.config import load_config
from rebike.data import DemandModel
from rebike.policies import build_policies, parse_models, run_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", action="append", help="NAME=PATH of a trained agent (repeatable)")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--demand", default=None, help="demand model .npz (default: eval.model_path = test month)")
    p.add_argument("--tag", default="", help="suffix for output files")
    args = p.parse_args()

    cfg = load_config(args.config)
    ec = cfg["eval"]
    n_ep = args.episodes or ec["episodes"]
    out = Path(ec["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    demand = DemandModel.load(Path(args.demand or ec["model_path"]))
    policies = build_policies(cfg["env"], demand, parse_models(args.model),
                              tuned_path=cfg["threshold_tuning"]["out"])

    rows = []
    for pol, env in policies:
        for ep in range(n_ep):          # same seeds -> same demand days for every policy
            r = run_episode(env, pol, ec["seed"] + ep)
            r.pop("hourly_served"); r.pop("hourly_demand")
            rows.append(dict(policy=pol.name, episode=ep, **r))
        print(f"[eval] {pol.name} done")
    df = pd.DataFrame(rows)
    sfx = f"_{args.tag}" if args.tag else ""
    df.to_csv(out / f"eval_episodes{sfx}.csv", index=False)

    cols = ["ret", "service_rate", "served", "lost", "km", "moved", "failed_docks"]
    summary = df.groupby("policy", sort=False)[cols].agg(["mean", "std"])
    summary.to_csv(out / f"eval_summary{sfx}.csv")
    print(summary.round(3).to_string())

    order = [p.name for p, _ in policies]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    for a, c, t in zip(ax, ["ret", "service_rate", "lost", "km"],
                       ["Episode return", "Service rate", "Lost trips / day", "Truck km / day"]):
        a.boxplot([df.loc[df.policy == n, c] for n in order], showfliers=False)
        a.set_xticks(range(1, len(order) + 1), [n.replace("_", "\n") for n in order], fontsize=8)
        a.set_title(t)
        a.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Held-out evaluation ({n_ep} simulated weekdays, {len(demand.station_ids)} stations)")
    fig.tight_layout()
    fig.savefig(out / f"eval_comparison{sfx}.png", dpi=150)
    print(f"[eval] wrote {out}/eval_summary{sfx}.csv and eval_comparison{sfx}.png")


if __name__ == "__main__":
    main()
