"""Shared helpers to build every policy (baselines and trained agents) for evaluation.

Each policy gets its own environment instance so agents trained with different
observation sets (e.g. with / without `extra_features`) can be compared on the
exact same simulated days: observations never consume randomness, so the same
seed gives the same demand for every policy.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .baselines import NoRebalancing, ThresholdPolicy
from .data import DemandModel
from .env import EnvConfig, RebalancingEnv


class PPOPolicy:
    def __init__(self, path: str | Path, env: RebalancingEnv, name: str = "ppo"):
        self.name, self.env, self.path = name, env, str(path)
        from stable_baselines3 import PPO
        from sb3_contrib import MaskablePPO
        try:
            self.model = MaskablePPO.load(path, device="cpu")
        except Exception:
            self.model = PPO.load(path, device="cpu")
        self.masked = "Maskable" in type(self.model.policy).__name__

    def act(self, obs):
        if self.masked:
            a, _ = self.model.predict(obs, action_masks=self.env.action_masks(), deterministic=True)
        else:
            a, _ = self.model.predict(obs, deterministic=True)
        return int(a)


def spaces_of(path: str | Path) -> tuple[int, int]:
    from stable_baselines3 import PPO
    from sb3_contrib import MaskablePPO
    try:
        m = MaskablePPO.load(path, device="cpu")
    except Exception:
        m = PPO.load(path, device="cpu")
    return int(m.observation_space.shape[0]), int(m.action_space.n)


def obs_dim_of(path: str | Path) -> int:
    return spaces_of(path)[0]


def env_for_model(path, cfg: EnvConfig, demand: DemandModel) -> RebalancingEnv:
    """Environment whose observation layout matches the saved model."""
    n = len(demand.station_ids)
    dim, n_act = spaces_of(path)
    if dim == 6 * n + 3:
        cfg = replace(cfg, extra_features=True)
    elif dim == 4 * n + 3:
        cfg = replace(cfg, extra_features=False)
    else:
        raise ValueError(f"{path}: observation dim {dim} does not match {n} stations")
    if n_act == n * (len(cfg.target_fills) + 1):
        cfg = replace(cfg, action_mode="target")
    elif n_act == n * len(cfg.transfer_levels):
        cfg = replace(cfg, action_mode="amount")
    else:
        raise ValueError(f"{path}: {n_act} actions match neither action mode")
    return RebalancingEnv(cfg, demand)


def build_policies(cfg: EnvConfig, demand: DemandModel, models: dict[str, str],
                   tuned_path: str | Path | None = None, include_default_threshold=True):
    """Return [(policy, env)] for the baselines plus every named PPO model."""
    model_cfg = cfg
    cfg = replace(cfg, action_mode="amount")    # baselines speak the bike-amount action set
    base = RebalancingEnv(cfg, demand)
    out = [(NoRebalancing(base), base)]
    if include_default_threshold:
        env = RebalancingEnv(cfg, demand)
        out.append((ThresholdPolicy.from_params(env, {}, "threshold_default"), env))
    if tuned_path and Path(tuned_path).exists():
        env = RebalancingEnv(cfg, demand)
        params = json.loads(Path(tuned_path).read_text())["best"]
        out.append((ThresholdPolicy.from_params(env, params, "threshold_tuned"), env))
    for name, path in models.items():
        env = env_for_model(path, model_cfg, demand)
        out.append((PPOPolicy(path, env, name), env))
    return out


def run_episode(env: RebalancingEnv, policy, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    ret, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(policy.act(obs))
        ret += r
        done = term or trunc
    out = dict(info, ret=info["abs_ret"], train_ret=ret)   # ret = absolute reward (comparable)
    out["hourly_served"] = env.hourly_served.copy()
    out["hourly_demand"] = env.hourly_demand.copy()
    return out


def parse_models(specs: list[str] | None) -> dict[str, str]:
    """['ppo_v2=models/ppo_v2_best.zip', 'models/x.zip'] -> {name: path}."""
    models = {}
    for s in specs or []:
        name, _, path = s.partition("=")
        if not path:
            name, path = "ppo", s
        models[name] = path
    return models


__all__ = ["PPOPolicy", "build_policies", "run_episode", "parse_models", "env_for_model", "np"]
