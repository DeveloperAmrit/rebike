"""ReBike rebalancing environment (Gymnasium).

One episode = one weekday of service (default 06:00-22:00) at the N busiest
stations. Rider demand is sampled from the Citi Bike demand model; a single
truck repositions bikes.

Decision epochs are event-driven (semi-MDP): each action sends the truck to a
station and moves bikes there; the clock then advances by the travel +
handling time, simulating rider demand in `tick_min` steps while the truck drives.

Initial state ("warm" mode): the system starts at `fleet_fill` of capacity,
one full day of demand is simulated without any truck, in-flight riders finish
their trips, and an overnight crew restores a fraction `overnight_rebalance`
of each station's deviation from the fleet average. Stations therefore begin
the morning in the imbalanced state that commuting patterns actually produce.

Observation (float32, dim 4N + 3)
    inv / cap                 (N)  fill level of every station
    one-hot truck location    (N)
    truck load / truck cap    (1)
    sin, cos of time-of-day   (2)
    expected net inflow over next hour / cap   (N)
    bikes already in transit to station (next 30 min) / cap   (N)

Action: Discrete(N * K), a = station * K + level
    station = where the truck drives next, level indexes `transfer_levels`
    (+x = drop x bikes from truck, -x = pick up x bikes; clipped to feasibility)
    `action_masks()` hides transfers that cannot be executed (for MaskablePPO).

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
    model_path: str = "data/processed/demand_train.npz"
    start_hour: int = 6
    end_hour: int = 22
    tick_min: int = 5
    truck_capacity: int = 20
    truck_speed_kmh: float = 18.0
    handling_min: float = 3.0
    transfer_levels: list = field(default_factory=lambda: [-10, -5, -2, 0, 2, 5, 10])
    init_mode: str = "warm"            # "warm" | "uniform"
    fleet_fill: float = 0.4            # bikes in system / total docks
    overnight_rebalance: float = 0.3   # warm mode: share of imbalance fixed overnight
    init_fill_low: float = 0.3         # uniform mode
    init_fill_high: float = 0.7        # uniform mode
    demand_noise_sigma: float = 0.15   # day-level log-normal demand multiplier
    w_served: float = 0.0
    w_lost: float = 1.0
    w_km: float = 0.2
    w_dock: float = 0.25


class RebalancingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None, model: DemandModel | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.m = model or DemandModel.load(Path(self.cfg.model_path))
        self.N = len(self.m.station_ids)
        self.cap = self.m.capacity.astype(np.int64)
        self.levels = np.asarray(self.cfg.transfer_levels, dtype=np.int64)
        self.K = len(self.levels)
        self.zero_level = int(np.argmin(np.abs(self.levels)))
        self.horizon = (self.cfg.end_hour - self.cfg.start_hour) * 60
        self.ride_ticks = np.maximum(1, np.round(self.m.ride_min / self.cfg.tick_min)).astype(np.int64)
        self.max_ride = int(self.ride_ticks.max()) + 1
        self.n_ticks = self.horizon // self.cfg.tick_min
        # stations ordered by distance, for redirecting riders who find a full dock
        self.nearest = np.argsort(self.m.dist_km, axis=1)
        # expected arrivals per hour: sum_i dep[h,i] * P[h,i,j]
        self.arr_rate = np.einsum("hi,hij->hj", self.m.dep_rate, self.m.od_prob)
        self.od_cdf = np.cumsum(self.m.od_prob, axis=-1)  # for vectorised destination sampling

        self.action_space = spaces.Discrete(self.N * self.K)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4 * self.N + 3,), dtype=np.float32)

    # ------------------------------------------------------------------ #
    def encode(self, station: int, level: int) -> int:
        return int(station) * self.K + int(level)

    def decode(self, action) -> tuple[int, int]:
        a = np.asarray(action)
        if a.ndim == 0:
            return int(a) // self.K, int(a) % self.K
        return int(a[0]), int(a[1])

    def action_masks(self) -> np.ndarray:
        """Feasible (station, transfer) pairs.

        Drop-offs: every level <= feasible amount f = min(load, free docks), plus the
        next larger level (which then moves exactly f). Pick-ups symmetric. Driving
        without transferring (level 0) is always allowed.
        """
        mask = np.zeros((self.N, self.K), dtype=bool)
        mask[:, self.zero_level] = True
        drop_f = np.minimum(self.truck_load, self.cap - self.inv)                 # (N,)
        pick_f = np.minimum(self.cfg.truck_capacity - self.truck_load, self.inv)  # (N,)
        pos = np.sort(self.levels[self.levels > 0])
        neg = np.sort(-self.levels[self.levels < 0])
        for amounts, f in ((pos, drop_f), (neg, pick_f)):
            sign = 1 if amounts is pos else -1
            for n, x in enumerate(amounts):
                prev = amounts[n - 1] if n else 0
                ok = (f >= x) | ((f > prev) & (f < x))
                mask[:, int(np.nonzero(self.levels == sign * x)[0][0])] = ok
        return mask.reshape(-1)

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.day_scale = float(np.exp(rng.normal(0.0, self.cfg.demand_noise_sigma)))
        if self.cfg.init_mode == "warm":
            self.inv = self._warm_start()
        else:
            fill = rng.uniform(self.cfg.init_fill_low, self.cfg.init_fill_high, self.N)
            self.inv = np.round(fill * self.cap).astype(np.int64)
        self.truck_pos = int(rng.integers(self.N))
        self.truck_load = self.cfg.truck_capacity // 2
        self.t = 0  # minutes since start_hour
        self.in_transit = np.zeros((self.n_ticks + self.max_ride + 1, self.N), dtype=np.int64)
        self.stats = dict(served=0, lost=0, km=0.0, moved=0, failed_docks=0, demand=0)
        return self._obs(), self._info()

    def _warm_start(self) -> np.ndarray:
        """Simulate the previous day without rebalancing, then a partial overnight fix."""
        self.inv = np.round(self.cfg.fleet_fill * self.cap).astype(np.int64)
        self.t = 0
        self.in_transit = np.zeros((self.n_ticks + self.max_ride + 1, self.N), dtype=np.int64)
        self.stats = dict(served=0, lost=0, km=0.0, moved=0, failed_docks=0, demand=0)
        while self.t < self.horizon:
            self._simulate_tick()
        # riders still on the road finish their trips
        self._dock(self.in_transit[self.n_ticks:].sum(0))
        # overnight crew moves a share of bikes back toward the fleet average fill
        total = self.inv.sum()
        target = total * self.cap / self.cap.sum()
        inv = np.round(self.inv + self.cfg.overnight_rebalance * (target - self.inv)).astype(np.int64)
        inv = np.clip(inv, 0, self.cap)
        diff = total - inv.sum()  # keep the number of bikes constant after rounding
        order = np.argsort(inv / self.cap) if diff > 0 else np.argsort(-inv / self.cap)
        for j in order:
            if diff == 0:
                break
            step = int(np.clip(diff, -inv[j], self.cap[j] - inv[j]))
            inv[j] += step
            diff -= step
        return inv

    # ------------------------------------------------------------------ #
    def step(self, action):
        target, lvl = self.decode(action)
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
    def _dock(self, arrivals: np.ndarray) -> int:
        """Dock returning bikes; overflow goes to the nearest station with room."""
        dock = np.minimum(arrivals, self.cap - self.inv)
        self.inv += dock
        spill_vec = arrivals - dock
        failed = int(spill_vec.sum())
        for j in np.nonzero(spill_vec)[0]:
            spill = int(spill_vec[j])
            for alt in self.nearest[j][1:]:
                put = min(spill, self.cap[alt] - self.inv[alt])
                self.inv[alt] += put
                spill -= put
                if spill == 0:
                    break
        return failed

    def _simulate_tick(self):
        rng = self.np_random
        k = self.t // self.cfg.tick_min
        h = (self.cfg.start_hour + self.t // 60) % 24

        # 1) returns
        failed = self._dock(self.in_transit[k])

        # 2) rentals: Poisson demand, served only if a bike is available
        lam = self.m.dep_rate[h] * self.day_scale * self.cfg.tick_min / 60.0
        demand = rng.poisson(lam)
        served_vec = np.minimum(demand, self.inv)
        self.inv -= served_vec
        lost = int((demand - served_vec).sum())
        served = int(served_vec.sum())

        # 3) route served riders to destinations
        if served:
            origins = np.repeat(np.arange(self.N), served_vec)
            u = rng.random(served)[:, None]
            dests = np.minimum((u > self.od_cdf[h, origins]).sum(1), self.N - 1)
            np.add.at(self.in_transit, (k + self.ride_ticks[origins, dests], dests), 1)

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
