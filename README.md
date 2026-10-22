# ReBike — RL for Bike-Share Rebalancing

AI Course · Assignment 2 (Real-Life Use Cases of RL)

**Team:** Amrit Kumar Yadav (B24CM1008) · Raj Kor (B24CM1040)

Bike-share stations that run empty or full lose rides. ReBike models truck-based
rebalancing as a Markov decision process on a demand simulator built from real
Citi Bike trip records, trains a **PPO** agent, and compares it against
**no rebalancing** and a **threshold-based** rebalancing policy.

## RL formulation

| | |
|---|---|
| **State** `s ∈ R^(4N+3)` | station fill levels `inv/cap` (N), one-hot truck location (N), truck load (1), `sin/cos` time of day (2), expected net inflow next hour (N), bikes in transit arriving in ≤30 min (N). Default N = 30 → 123 dims |
| **Action** `MultiDiscrete([N, K])` | next station to visit × bikes to move from `{-10,-5,-2,0,+2,+5,+10}` (− pick up, + drop off; clipped to truck/dock limits) |
| **Reward** | `r = w_s·served − w_l·lost − w_km·km − w_d·failed_docks` per decision (default `1, 0, 0.5, 0.1`) |
| **Transition** | event-driven (semi-MDP): the clock advances by truck travel + handling time; rider demand is simulated in 5-minute ticks meanwhile |
| **Episode** | one weekday, 06:00–22:00 |

### Simulator
Built by `src/rebike/data.py` from Citi Bike monthly trip archives:
- the N busiest stations (trips with both ends inside the set),
- hourly departure rates `λ[h,i]` (Poisson rentals, scaled by a log-normal day factor),
- hourly origin–destination probabilities `P[h,i,j]`,
- median ride times between stations (riders arrive after the ride),
- full docks redirect riders to the nearest free dock (counted as failed docks),
- empty stations lose the rental (lost trip).

## Repository layout

```
configs/default.yaml       all data / env / PPO / eval settings
src/rebike/data.py         download trips, build demand model (.npz)
src/rebike/env.py          Gymnasium environment
src/rebike/baselines.py    no-rebalancing and threshold policies
scripts/prepare_data.py    data pipeline entry point
scripts/train_ppo.py       PPO training (Stable-Baselines3)
scripts/evaluate.py        policy comparison + plots
tests/test_env.py          environment smoke tests
documentation/             assignment brief, proposal, team
```

## Installation

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Reproducing

```bash
# 1. data: downloads Jersey City/Hoboken trips (Jun–Oct 2024) and builds the demand model
python scripts/prepare_data.py --config configs/default.yaml

# 2. tests
pytest -q

# 3. baselines only
python scripts/evaluate.py --config configs/default.yaml

# 4. train PPO (add --subproc to run envs in parallel processes)
python scripts/train_ppo.py --config configs/default.yaml

# 5. compare PPO vs baselines -> results/eval_summary.csv, results/eval_comparison.png
python scripts/evaluate.py --config configs/default.yaml --model models/ppo_rebike.zip

# training curves
tensorboard --logdir results/tb
```

To use the NYC system instead, set `data.region: NYC` and the months you want in
`configs/default.yaml` (each NYC month is ~1 GB).

## Status

- [x] Data pipeline and demand model
- [x] Simulator environment + tests
- [x] Baselines (no rebalancing, threshold)
- [x] PPO training / evaluation scripts
- [ ] Hyperparameter tuning and full training run
- [ ] Results, graphs and 2-page report
