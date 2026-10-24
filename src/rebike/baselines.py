"""Non-learning reference policies used only for comparison against PPO."""
from __future__ import annotations

import numpy as np

from .env import RebalancingEnv


class NoRebalancing:
    """Truck stays where it is and never moves bikes."""
    name = "no_rebalancing"

    def __init__(self, env: RebalancingEnv):
        self.env = env

    def act(self, obs):
        return self.env.encode(self.env.truck_pos, self.env.zero_level)


class ThresholdPolicy:
    """Classic threshold rebalancing.

    If the truck carries bikes, serve the station below `low` fill with the best
    deficit/distance ratio; otherwise collect from the station above `high`
    fill with the best surplus/distance ratio. Move bikes toward `target` fill.
    """
    name = "threshold"

    @classmethod
    def from_params(cls, env, params: dict, name: str = "threshold"):
        pol = cls(env, **params)
        pol.name = name
        return pol

    def __init__(self, env: RebalancingEnv, low=0.2, high=0.8, target=0.5):
        self.env, self.low, self.high, self.target = env, low, high, target

    def _level(self, want: int) -> int:
        return int(np.argmin(np.abs(self.env.levels - want)))

    def act(self, obs):
        e = self.env
        fill = e.inv / e.cap
        dist = e.m.dist_km[e.truck_pos]
        goal = np.round(self.target * e.cap).astype(int)

        empty = np.nonzero(fill < self.low)[0]
        full = np.nonzero(fill > self.high)[0]

        if e.truck_load >= 2 and len(empty):
            score = (goal[empty] - e.inv[empty]) / (1.0 + dist[empty])
            j = int(empty[np.argmax(score)])
            return e.encode(j, self._level(min(goal[j] - e.inv[j], e.truck_load)))

        space = e.cfg.truck_capacity - e.truck_load
        if space >= 2 and len(full):
            score = (e.inv[full] - goal[full]) / (1.0 + dist[full])
            j = int(full[np.argmax(score)])
            return e.encode(j, self._level(-min(e.inv[j] - goal[j], space)))

        return e.encode(e.truck_pos, e.zero_level)
