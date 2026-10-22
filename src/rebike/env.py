"""ReBike rebalancing environment (Gymnasium).

One episode = one weekday of service (default 06:00-22:00) at the N busiest
stations. Rider demand is sampled from the Citi Bike demand model; a single
truck repositions bikes.

Decision epochs are event-driven (semi-MDP): each action sends the truck to a
station and moves bikes there; the clock then advances by the travel +
handling time, simulating rider demand minute-by-minute (in `tick_min` steps)
while the truck is driving.

Observation (float32, dim 4N + 3)
    inv / cap                 (N)  fill level of every station
    one-hot truck location    (N)
    truck load / truck cap    (1)
    sin, cos of time-of-day   (2)
    expected net inflow over next hour / cap   (N)
    bikes already in transit to station (next 30 min) / cap   (N)

Action: MultiDiscrete([N, K])
    a[0] target station, a[1] index into `transfer_levels`
    (+x = drop x bikes from truck, -x = pick up x bikes; clipped to feasibility)

Reward per decision
    r = w_served * served - w_lost * lost - w_km * km_driven - w_dock * failed_docks
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .data import DemandModel


@dataclass
class EnvConfig:
    model_path: str = "data/processed/demand_JC.npz"
    start_hour: int = 6
    end_hour: int = 22
    tick_min: int = 5
    truck_capacity: int = 20
    truck_speed_kmh: float = 18.0
    handling_min: float = 3.0
    transfer_levels: list = field(default_factory=lambda: [-10, -5, -2, 0, 2, 5, 10])
    init_fill_low: float = 0.3
    init_fill_high: float = 0.7
    demand_noise_sigma: float = 0.15   # day-level log-normal demand multiplier
    w_served: float = 1.0
    w_lost: float = 0.0
    w_km: float = 0.5
    w_dock: float = 0.1


class RebalancingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None, model: DemandModel | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.m = model or DemandModel.load(Path(self.cfg.model_path))
        self.N = len(self.m.station_ids)
        self.cap = self.m.capacity.astype(np.int64)
        self.levels = np.asarray(self.cfg.transfer_levels, dtype=np.int64)
        self.horizon = (self.cfg.end_hour - self.cfg.start_hour) * 60
        self.ride_ticks = np.maximum(1, np.round(self.m.ride_min / self.cfg.tick_min)).astype(np.int64)
        self.max_ride = int(self.ride_ticks.max()) + 1
        self.n_ticks = self.horizon // self.cfg.tick_min
        # stations ordered by distance, for redirecting riders who find a full dock
        self.nearest = np.argsort(self.m.dist_km, axis=1)
        # expected arrivals per hour: sum_i dep[h,i] * P[h,i,j]
        self.arr_rate = np.einsum("hi,hij->hj", self.m.dep_rate, self.m.od_prob)

        self.action_space = spaces.MultiDiscrete([self.N, len(self.levels)])
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4 * self.N + 3,), dtype=np.float32)

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        fill = rng.uniform(self.cfg.init_fill_low, self.cfg.init_fill_high, self.N)
        self.inv = np.round(fill * self.cap).astype(np.int64)
        self.truck_pos = int(rng.integers(self.N))
        self.truck_load = self.cfg.truck_capacity // 2
        self.t = 0  # minutes since start_hour
        self.in_transit = np.zeros((self.n_ticks + self.max_ride + 1, self.N), dtype=np.int64)
        self.day_scale = float(np.exp(rng.normal(0.0, self.cfg.demand_noise_sigma)))
        self.stats = dict(served=0, lost=0, km=0.0, moved=0, failed_docks=0, demand=0)
        return self._obs(), self._info()

    # ------------------------------------------------------------------ #
    def step(self, action):
        target, lvl = int(action[0]), int(action[1])
        km = float(self.m.dist_km[self.truck_pos, target])
        minutes = km / self.cfg.truck_speed_kmh * 60 + self.cfg.handling_min
        ticks = max(1, int(np.ceil(minutes / self.cfg.tick_min)))

        served = lost = failed = 0
        for _ in range(ticks):
            if self.t >= self.horizon:
                break
            s, l, f = self._simulate_tick()
            served, lost, failed = served + s, lost + l, failed + f

        # truck arrives and transfers bikes
        self.truck_pos = target
        delta = int(self.levels[lvl])
        if delta > 0:
            moved = min(delta, self.truck_load, self.cap[target] - self.inv[target])
            self.inv[target] += moved
            self.truck_load -= moved
        elif delta < 0:
            moved = min(-delta, self.inv[target], self.cfg.truck_capacity - self.truck_load)
            self.inv[target] -= moved
            self.truck_load += moved
        else:
            moved = 0

        self.stats["km"] += km
        self.stats["moved"] += int(moved)
        c = self.cfg
        reward = c.w_served * served - c.w_lost * lost - c.w_km * km - c.w_dock * failed
        terminated = self.t >= self.horizon
        return self._obs(), float(reward), terminated, False, self._info()

    # ------------------------------------------------------------------ #
    def _simulate_tick(self):
        rng = self.np_random
        k = self.t // self.cfg.tick_min
        h = (self.cfg.start_hour + self.t // 60) % 24

        # 1) returns: bikes finishing rides dock (overflow -> nearest free dock)
        failed = 0
        arrivals = self.in_transit[k]
        for j in np.nonzero(arrivals)[0]:
            n = int(arrivals[j])
            room = self.cap[j] - self.inv[j]
            dock = min(n, room)
            self.inv[j] += dock
            spill = n - dock
            if spill:
                failed += spill
                for alt in self.nearest[j][1:]:
                    r = self.cap[alt] - self.inv[alt]
                    put = min(spill, r)
                    self.inv[alt] += put
                    spill -= put
                    if spill == 0:
                        break

        # 2) rentals: Poisson demand, served only if a bike is available
        lam = self.m.dep_rate[h] * self.day_scale * self.cfg.tick_min / 60.0
        demand = rng.poisson(lam)
        served_vec = np.minimum(demand, self.inv)
        self.inv -= served_vec
        lost = int((demand - served_vec).sum())
        served = int(served_vec.sum())

        # 3) route served riders to destinations
        for i in np.nonzero(served_vec)[0]:
            dests = rng.choice(self.N, size=int(served_vec[i]), p=self.m.od_prob[h, i])
            for j in dests:
                self.in_transit[k + self.ride_ticks[i, j], j] += 1

        self.stats["served"] += served
        self.stats["lost"] += lost
        self.stats["failed_docks"] += failed
        self.stats["demand"] += int(demand.sum())
        self.t += self.cfg.tick_min
        return served, lost, failed

    # ------------------------------------------------------------------ #
    def _obs(self):
        k = min(self.t // self.cfg.tick_min, self.n_ticks)
        h = (self.cfg.start_hour + min(self.t, self.horizon - 1) // 60) % 24
        frac = (self.cfg.start_hour * 60 + self.t) / (24 * 60)
        pos = np.zeros(self.N)
        pos[self.truck_pos] = 1.0
        net = (self.arr_rate[h] - self.m.dep_rate[h]) * self.day_scale / self.cap
        soon = self.in_transit[k : k + max(1, 30 // self.cfg.tick_min)].sum(0) / self.cap
        return np.concatenate([
            self.inv / self.cap,
            pos,
            [self.truck_load / self.cfg.truck_capacity],
            [np.sin(2 * np.pi * frac), np.cos(2 * np.pi * frac)],
            net,
            soon,
        ]).astype(np.float32)

    def _info(self):
        d = dict(self.stats)
        d["service_rate"] = d["served"] / max(d["demand"], 1)
        d["time_min"] = self.t
        return d


def make_env(config: EnvConfig, model: DemandModel | None = None):
    def _f():
        return RebalancingEnv(config, model)
    return _f
