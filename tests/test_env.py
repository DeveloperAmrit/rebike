"""Smoke tests on a tiny synthetic demand model (no download needed)."""
import numpy as np
import pytest

from rebike.baselines import NoRebalancing, ThresholdPolicy
from rebike.data import DemandModel, haversine_km
from rebike.env import EnvConfig, RebalancingEnv


@pytest.fixture
def toy_model():
    rng = np.random.default_rng(0)
    N = 6
    coords = np.c_[40.72 + rng.uniform(0, 0.02, N), -74.04 + rng.uniform(0, 0.02, N)]
    dist = haversine_km(coords, coords) * 1.35
    dep = rng.uniform(0.5, 4.0, (24, N))
    od = rng.dirichlet(np.ones(N), (24, N))
    return DemandModel(
        station_ids=np.array([f"S{i}" for i in range(N)]), coords=coords,
        capacity=np.full(N, 15), dep_rate=dep, od_prob=od,
        ride_min=dist / 12 * 60 + 2, dist_km=dist, n_trips=1000, n_days=10,
    )


def test_spaces_and_episode(toy_model):
    env = RebalancingEnv(EnvConfig(), toy_model)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    done, steps = False, 0
    while not done:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(r) and obs.shape == env.observation_space.shape
        assert (env.inv >= 0).all() and (env.inv <= env.cap).all()
        assert 0 <= env.truck_load <= env.cfg.truck_capacity
        done = term or trunc
        steps += 1
    assert info["served"] + info["lost"] == info["demand"]
    assert steps > 10


def test_bike_conservation(toy_model):
    env = RebalancingEnv(EnvConfig(), toy_model)
    env.reset(seed=1)
    total = env.inv.sum() + env.truck_load
    pol = ThresholdPolicy(env)
    for _ in range(30):
        obs, *_ = env.step(pol.act(env._obs()))
    k = env.t // env.cfg.tick_min
    riding = env.in_transit[k:].sum()
    assert env.inv.sum() + env.truck_load + riding == total


@pytest.mark.parametrize("cls", [NoRebalancing, ThresholdPolicy])
def test_baselines_run(toy_model, cls):
    env = RebalancingEnv(EnvConfig(), toy_model)
    obs, _ = env.reset(seed=2)
    pol = cls(env)
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(pol.act(obs))
        done = term or trunc
    assert 0 <= info["service_rate"] <= 1


def test_warm_start_keeps_fleet_size(toy_model):
    cfg = EnvConfig(init_mode="warm", fleet_fill=0.4)
    env = RebalancingEnv(cfg, toy_model)
    expected = np.round(0.4 * env.cap).sum()
    for s in range(3):
        env.reset(seed=s)
        assert env.inv.sum() == expected
        assert (env.inv >= 0).all() and (env.inv <= env.cap).all()


def test_action_mask_matches_feasibility(toy_model):
    env = RebalancingEnv(EnvConfig(), toy_model)
    rng = np.random.default_rng(3)
    for s in range(20):
        env.reset(seed=s)
        env.truck_load = int(rng.integers(0, env.cfg.truck_capacity + 1))
        env.inv = rng.integers(0, env.cap + 1)
        mask = env.action_masks().reshape(env.N, env.K)
        assert mask[:, env.zero_level].all()
        for j in range(env.N):
            drop = min(env.truck_load, env.cap[j] - env.inv[j])
            pick = min(env.cfg.truck_capacity - env.truck_load, env.inv[j])
            for k, x in enumerate(env.levels):
                feasible = drop if x > 0 else pick if x < 0 else 1
                if mask[j, k] and x != 0:
                    assert feasible > 0           # an allowed transfer always moves bikes
                if feasible >= abs(x) and x != 0:
                    assert mask[j, k]             # every exactly-feasible transfer allowed
