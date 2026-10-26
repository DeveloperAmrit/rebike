"""Plot PPO learning curves from VecMonitor logs.

    python scripts/plot_training.py --run ppo_v4 --run ppo_v4_nomask --run ppo_v2
Writes results/training_curve.png

Uses the absolute per-episode metrics logged by the monitor (service rate and
lost trips per day), which are comparable across reward modes; the raw episode
reward differs between "absolute" and "counterfactual" runs. Dashed lines are the
baselines on the same (training) demand, evaluated on validation days if
results/baselines_train.csv exists.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LABELS = {"ppo_v4": "PPO v4 (ours)", "ppo_v4_nomask": "v4 without mask", "ppo_v2": "v2 (amount actions, abs. reward)",
          "ppo_v3": "v3 (target actions, abs. reward)", "ppo_rebike": "v1"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", help="run name; reads results/<run>_monitor.monitor.csv")
    p.add_argument("--baselines", default="results/baselines_train.csv")
    p.add_argument("--out", default="results/training_curve.png")
    p.add_argument("--window", type=int, default=200)
    args = p.parse_args()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for i, run in enumerate(args.run or ["ppo_v4"]):
        f = Path(f"results/{run}_monitor.monitor.csv")
        if not f.exists():
            print(f"[plot] skip {run}: {f} missing")
            continue
        df = pd.read_csv(f, skiprows=1)
        df["steps"] = df["l"].cumsum()
        roll = df.rolling(args.window, min_periods=20).mean()
        lw = 1.8 if i == 0 else 1.1
        ax[0].plot(df["steps"], roll["service_rate"], color=f"C{i}", lw=lw, label=LABELS.get(run, run))
        ax[1].plot(df["steps"], roll["lost"], color=f"C{i}", lw=lw)
    if Path(args.baselines).exists():
        b = pd.read_csv(args.baselines)
        for j, row in b.iterrows():
            c = {"no_rebalancing": "0.5", "threshold_tuned": "k"}.get(row["policy"], "0.3")
            ax[0].axhline(row["service_rate"], ls="--", color=c, lw=1, label=row["policy"])
            ax[1].axhline(row["lost"], ls="--", color=c, lw=1)
    ax[0].set(title="Service rate (training demand)", xlabel="environment steps")
    ax[1].set(title="Lost trips per day (training demand)", xlabel="environment steps")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
