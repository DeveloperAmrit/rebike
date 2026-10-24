"""Plot PPO learning curves from the VecMonitor log.

    python scripts/plot_training.py --monitor results/ppo_rebike_monitor.monitor.csv
Writes results/training_curve.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--monitor", default="results/ppo_rebike_monitor.monitor.csv")
    p.add_argument("--baselines", default="results/eval_summary.csv",
                   help="optional: draw baseline returns as reference lines")
    p.add_argument("--out", default="results/training_curve.png")
    p.add_argument("--window", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.monitor, skiprows=1)
    df["steps"] = df["l"].cumsum()
    roll = df.rolling(args.window, min_periods=1).mean()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(df["steps"], roll["r"], color="C0", label="PPO (train demand)")
    ax[1].plot(df["steps"], roll["service_rate"], color="C0")
    if Path(args.baselines).exists():
        b = pd.read_csv(args.baselines, header=[0, 1], index_col=0)
        for i, (name, row) in enumerate(b.iterrows()):
            if name == "ppo":
                continue
            ax[0].axhline(row[("ret", "mean")], ls="--", color=f"C{i + 1}", label=f"{name} (test)")
            ax[1].axhline(row[("service_rate", "mean")], ls="--", color=f"C{i + 1}")
    ax[0].set(title="Episode return", xlabel="environment steps")
    ax[1].set(title="Service rate", xlabel="environment steps")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[plot] wrote {args.out}")


if __name__ == "__main__":
    main()
