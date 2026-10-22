"""Compare PPO against the no-rebalancing and threshold baselines.

    python scripts/evaluate.py --config configs/default.yaml --model models/ppo_rebike.zip
Writes results/eval_summary.csv, results/eval_episodes.csv and results/eval_comparison.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rebike.baselines import NoRebalancing, ThresholdPolicy
from rebike.config import load_config
from rebike.data import DemandModel
from rebike.env import RebalancingEnv


class PPOPolicy:
    name = "ppo"

    def __init__(self, path):
        from stable_baselines3 import PPO
        self.model = PPO.load(path, device="cpu")

    def act(self, obs):
        a, _ = self.model.predict(obs, deterministic=True)
        return a


def run_episode(env, policy, seed):
    obs, info = env.reset(seed=seed)
    ret, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(policy.act(obs))
        ret += r
        done = term or trunc
    return dict(info, ret=ret)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", default=None, help="path to trained PPO .zip (optional)")
    p.add_argument("--episodes", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    ec = cfg["eval"]
    n_ep = args.episodes or ec["episodes"]
    out = Path(ec["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    demand = DemandModel.load(Path(cfg["env"].model_path))
    env = RebalancingEnv(cfg["env"], demand)
    policies = [NoRebalancing(env), ThresholdPolicy(env)]
    if args.model:
        policies.append(PPOPolicy(args.model))

    rows = []
    for pol in policies:
        for ep in range(n_ep):          # same seeds -> same demand days for every policy
            rows.append(dict(policy=pol.name, episode=ep, **run_episode(env, pol, ec["seed"] + ep)))
        print(f"[eval] {pol.name} done")
    df = pd.DataFrame(rows)
    df.to_csv(out / "eval_episodes.csv", index=False)

    cols = ["ret", "service_rate", "served", "lost", "km", "moved", "failed_docks"]
    summary = df.groupby("policy", sort=False)[cols].agg(["mean", "std"])
    summary.to_csv(out / "eval_summary.csv")
    print(summary.round(3).to_string())

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    for a, c, t in zip(ax, ["service_rate", "lost", "km"],
                       ["Service rate", "Lost trips / day", "Truck km / day"]):
        df.boxplot(column=c, by="policy", ax=a, grid=False)
        a.set_title(t); a.set_xlabel("")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(out / "eval_comparison.png", dpi=150)
    print(f"[eval] wrote {out}/eval_summary.csv and eval_comparison.png")


if __name__ == "__main__":
    main()
