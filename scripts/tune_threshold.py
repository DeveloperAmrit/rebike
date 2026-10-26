"""Grid-search the threshold baseline on the *training* demand model.

    python scripts/tune_threshold.py --config configs/default.yaml

Selects (low, high, target) by mean episode return so that PPO is compared
against the strongest version of the threshold policy, not a weak default.
Writes results/threshold_tuned.json (all grid points + best).
"""
import argparse
import itertools
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from rebike.baselines import ThresholdPolicy
from rebike.config import load_config
from rebike.data import DemandModel
from rebike.env import RebalancingEnv


def evaluate(env, params, seeds):
    rets, service = [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        pol = ThresholdPolicy(env, **params)
        ret, done = 0.0, False
        while not done:
            obs, r, term, trunc, info = env.step(pol.act(obs))
            ret += r
            done = term or trunc
        rets.append(info["abs_ret"])   # absolute reward, comparable across reward modes
        service.append(info["service_rate"])
    return float(np.mean(rets)), float(np.mean(service))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    tc = cfg["threshold_tuning"]

    env_cfg = replace(cfg["env"], action_mode="amount")   # threshold uses bike-amount actions
    env = RebalancingEnv(env_cfg, DemandModel.load(Path(cfg["env"].model_path)))
    seeds = [tc["seed"] + i for i in range(tc["episodes"])]
    grid = [dict(low=l, high=h, target=t)
            for l, h, t in itertools.product(tc["low"], tc["high"], tc["target"]) if l < t < h]

    results = []
    for n, params in enumerate(grid, 1):
        ret, sr = evaluate(env, params, seeds)
        results.append(dict(params, ret=ret, service_rate=sr))
        print(f"[tune] {n:3d}/{len(grid)} {params} return={ret:8.1f} service={sr:.3f}")

    best = max(results, key=lambda r: r["ret"])
    out = Path(tc["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        best={k: best[k] for k in ("low", "high", "target")},
        best_return=best["ret"], best_service_rate=best["service_rate"],
        episodes=tc["episodes"], grid=results), indent=2))
    print(f"[tune] best {best} -> {out}")


if __name__ == "__main__":
    main()
