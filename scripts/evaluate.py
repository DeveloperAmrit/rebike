"""Compare PPO against the baselines on held-out (test-month) demand.

    python scripts/evaluate.py --config configs/default.yaml --model models/ppo_rebike.zip

Every policy is run on the same seeded days (same demand, same morning state).
Policies: no rebalancing, threshold with default settings, threshold with tuned
settings (results/threshold_tuned.json, if present), and the trained PPO agent.
Writes results/eval_summary.csv, results/eval_episodes.csv, results/eval_comparison.png
"""
import argparse
import json
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

    def __init__(self, path, env):
        from sb3_contrib import MaskablePPO
        self.model = MaskablePPO.load(path, device="cpu")
        self.env = env

    def act(self, obs):
        a, _ = self.model.predict(obs, action_masks=self.env.action_masks(), deterministic=True)
        return int(a)


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
    p.add_argument("--model", default=None, help="trained PPO .zip (optional)")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--demand", default=None, help="demand model .npz (default: eval.model_path = test month)")
    args = p.parse_args()

    cfg = load_config(args.config)
    ec = cfg["eval"]
    n_ep = args.episodes or ec["episodes"]
    out = Path(ec["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    demand = DemandModel.load(Path(args.demand or ec["model_path"]))
    env = RebalancingEnv(cfg["env"], demand)
    policies = [NoRebalancing(env), ThresholdPolicy.from_params(env, {}, "threshold_default")]
    tuned = Path(cfg["threshold_tuning"]["out"])
    if tuned.exists():
        params = json.loads(tuned.read_text())["best"]
        policies.append(ThresholdPolicy.from_params(env, params, "threshold_tuned"))
    if args.model:
        policies.append(PPOPolicy(args.model, env))

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

    order = [p.name for p in policies]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    for a, c, t in zip(ax, ["ret", "service_rate", "lost", "km"],
                       ["Episode return", "Service rate", "Lost trips / day", "Truck km / day"]):
        a.boxplot([df.loc[df.policy == n, c] for n in order], showfliers=False)
        a.set_xticks(range(1, len(order) + 1), [n.replace("_", "\n") for n in order], fontsize=8)
        a.set_title(t)
        a.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Held-out evaluation ({n_ep} simulated weekdays, {len(demand.station_ids)} stations)")
    fig.tight_layout()
    fig.savefig(out / "eval_comparison.png", dpi=150)
    print(f"[eval] wrote {out}/eval_summary.csv and eval_comparison.png")


if __name__ == "__main__":
    main()
