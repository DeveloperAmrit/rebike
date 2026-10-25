"""Extra analyses for the report (held-out test demand).

    python scripts/analysis.py hourly     --model ppo_v2=models/ppo_v2_best.zip
    python scripts/analysis.py robustness --model ppo_v2=models/ppo_v2_best.zip

hourly      service rate per hour of day, averaged over the evaluation days
            -> results/hourly_service.csv, results/hourly_service.png
robustness  all policies under scaled demand (0.8x .. 1.3x of October demand),
            agents are NOT retrained -> results/robustness.csv, results/robustness.png
"""
import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rebike.config import load_config
from rebike.data import DemandModel
from rebike.policies import build_policies, parse_models, run_episode

STYLE = {"no_rebalancing": ("C1", "--"), "threshold_tuned": ("C3", "-."), "threshold_default": ("C2", ":")}


def hourly(cfg, demand, models, episodes, seed, out):
    pols = build_policies(cfg["env"], demand, models, cfg["threshold_tuning"]["out"],
                          include_default_threshold=False)
    start = cfg["env"].start_hour
    rows = []
    for pol, env in pols:
        served = demand_ = 0
        for ep in range(episodes):
            r = run_episode(env, pol, seed + ep)
            served = served + r["hourly_served"]
            demand_ = demand_ + r["hourly_demand"]
        for h, (s, d) in enumerate(zip(served, demand_)):
            rows.append(dict(policy=pol.name, hour=start + h, served=int(s), demand=int(d),
                             service_rate=s / max(d, 1)))
        print(f"[hourly] {pol.name} done")
    df = pd.DataFrame(rows)
    df.to_csv(out / "hourly_service.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for i, (name, g) in enumerate(df.groupby("policy", sort=False)):
        c, ls = STYLE.get(name, (f"C{i}", "-"))
        ax[0].plot(g["hour"], g["service_rate"], ls, color=c, marker="o", ms=3, label=name)
        ax[1].plot(g["hour"], (g["demand"] - g["served"]) / episodes, ls, color=c, marker="o", ms=3)
    ax[0].set(title="Service rate by hour", xlabel="hour of day")
    ax[1].set(title="Lost trips per hour (mean per day)", xlabel="hour of day")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "hourly_service.png", dpi=150)
    print(f"[hourly] wrote {out}/hourly_service.png")


def robustness(cfg, demand, models, episodes, seed, out, scales):
    rows = []
    for sc in scales:
        env_cfg = replace(cfg["env"], demand_scale=sc)
        pols = build_policies(env_cfg, demand, models, cfg["threshold_tuning"]["out"],
                              include_default_threshold=False)
        for pol, env in pols:
            for ep in range(episodes):
                r = run_episode(env, pol, seed + ep)
                rows.append(dict(policy=pol.name, scale=sc, ret=r["ret"],
                                 service_rate=r["service_rate"], lost=r["lost"], km=r["km"]))
        print(f"[robustness] demand x{sc} done")
    df = pd.DataFrame(rows)
    df.to_csv(out / "robustness.csv", index=False)
    agg = df.groupby(["policy", "scale"], sort=False).mean(numeric_only=True).reset_index()

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for i, (name, g) in enumerate(agg.groupby("policy", sort=False)):
        c, ls = STYLE.get(name, (f"C{i}", "-"))
        for a, col in zip(ax, ["ret", "service_rate", "km"]):
            a.plot(g["scale"], g[col], ls, color=c, marker="o", ms=4, label=name)
    for a, t in zip(ax, ["Episode return", "Service rate", "Truck km / day"]):
        a.set(title=t, xlabel="demand multiplier (1.0 = October 2024)")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "robustness.png", dpi=150)
    print(f"[robustness] wrote {out}/robustness.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["hourly", "robustness"])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", action="append", help="NAME=PATH of a trained agent (repeatable)")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--scales", default="0.8,0.9,1.0,1.1,1.2,1.3")
    args = p.parse_args()

    cfg = load_config(args.config)
    ec = cfg["eval"]
    out = Path(ec["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    demand = DemandModel.load(Path(ec["model_path"]))
    models = parse_models(args.model)
    if args.what == "hourly":
        hourly(cfg, demand, models, args.episodes or ec["episodes"], ec["seed"], out)
    else:
        scales = [float(s) for s in args.scales.split(",")]
        robustness(cfg, demand, models, args.episodes or 15, ec["seed"], out, scales)


if __name__ == "__main__":
    main()
