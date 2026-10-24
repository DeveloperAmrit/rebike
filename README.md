# ReBike — RL for Bike-Share Rebalancing

AI Course · Assignment 2 (Real-Life Use Cases of RL)

**Team:** Amrit Kumar Yadav (B24CM1008) · Raj Kor (B24CM1040)

Bike-share stations that run empty or full lose rides. ReBike models truck-based
rebalancing as a Markov decision process on a demand simulator built from real
Citi Bike trip records, trains a **PPO** agent (with invalid-action masking), and compares it against
**no rebalancing** and a **tuned threshold-based** rebalancing policy on held-out days.

## RL formulation

| | |
|---|---|
| **State** `s ∈ R^(4N+3)` | station fill levels `inv/cap` (N), one-hot truck location (N), truck load (1), `sin/cos` time of day (2), expected net inflow next hour (N), bikes in transit arriving in ≤30 min (N). Default N = 60 → 243 dims |
| **Action** `Discrete(N·K)` | next station to visit × bikes to move from `{-10,-5,-2,0,+2,+5,+10}` (− pick up, + drop off) = 420 actions. Infeasible transfers are **masked** (`env.action_masks()`, MaskablePPO) |
| **Reward** | `r = w_s·served − w_l·lost − w_km·km − w_d·failed_docks` per decision (default `0, 1, 0.2, 0.25`) |
| **Transition** | event-driven (semi-MDP): the clock advances by truck travel + handling time; rider demand is simulated in 5-minute ticks meanwhile |
| **Episode** | one weekday, 06:00–22:00 |
| **Initial state** | *warm start*: fleet at 40% of docks, one simulated day without rebalancing, then an overnight crew fixes 30% of each station's imbalance — so mornings start as unbalanced as commuting makes them |

**Reward design.** Total demand is exogenous (the agent cannot change it), so
`−lost` ranks policies exactly like `+served` but removes the demand noise from the
learning signal. The km cost (0.2 lost-trip equivalents per km) reflects a single
ride (~$4.79) vs. truck operating cost (~$1/km). Failed docks (rider finds a full
station and is redirected) cost a quarter of a lost trip.

### Simulator
Built by `src/rebike/data.py` from Citi Bike monthly trip archives:
- the N = 60 busiest Jersey City / Hoboken stations (trips with both ends inside the set),
- real dock counts from the Citi Bike GBFS feed (cached in `data/dock_capacity.json`),
- hourly departure rates `λ[h,i]` (Poisson rentals, scaled by a log-normal day factor),
- hourly origin–destination probabilities `P[h,i,j]`,
- median ride times between stations (riders arrive after the ride),
- full docks redirect riders to the nearest free dock (counted as failed docks),
- empty stations lose the rental (lost trip).

**Train / test split.** The demand model used for training is built from
Jun–Sep 2024 weekdays; all reported results use a separate model built from
**October 2024** (held out) on the same stations.

**Difficulty calibration.** With 30 stations and random starting fill, 95–97% of
trips were served with no truck at all, leaving nothing to learn. With 60
stations, real dock counts, a 40% fleet and the warm start, no rebalancing
serves ≈ 87% and the tuned threshold policy ≈ 91% of trips on held-out days.

## Repository layout

```
configs/default.yaml       all data / env / PPO / eval settings
src/rebike/data.py         download trips, build demand model (.npz)
src/rebike/env.py          Gymnasium environment
src/rebike/baselines.py    no-rebalancing and threshold policies
scripts/prepare_data.py    data pipeline entry point (train + test demand models)
scripts/tune_threshold.py  grid search for the threshold baseline
scripts/train_ppo.py       Maskable PPO training (sb3-contrib)
scripts/evaluate.py        held-out policy comparison + plots
scripts/plot_training.py   learning curves
tests/test_env.py          environment tests (conservation, masks, warm start)
data/dock_capacity.json    cached GBFS dock counts
models/ppo_rebike_best.zip selected trained agent (750k-step checkpoint)
models/ppo_rebike.zip      final agent after 1.5M steps
results/                   tuning, training and evaluation outputs
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
# 1. data: downloads Jersey City/Hoboken trips (Jun–Oct 2024), builds train + test demand models
python scripts/prepare_data.py --config configs/default.yaml

# 2. tests
pytest -q

# 3. tune the threshold baseline on training demand -> results/threshold_tuned.json  (~30 s)
python scripts/tune_threshold.py --config configs/default.yaml

# 4. train PPO, 1.5M steps (~55 min on 2 CPU cores; --subproc helps on 4+ cores)
python scripts/train_ppo.py --config configs/default.yaml --subproc
#    model selection: the 750k checkpoint scored best on training-demand validation days
cp models/checkpoints/ppo_rebike_750000_steps.zip models/ppo_rebike_best.zip

# 5. evaluate all policies on 30 held-out October days
python scripts/evaluate.py --config configs/default.yaml --model models/ppo_rebike_best.zip

# 6. learning curve -> results/training_curve.png
python scripts/plot_training.py
tensorboard --logdir results/tb       # optional

# evaluate the committed agent without retraining: run steps 1, 3 and 5
```

To use the NYC system instead, set `data.region: NYC` and the months you want in
`configs/default.yaml` (each NYC month is ~1 GB).

## Results (first full run)

Held-out October 2024 demand, 30 simulated weekdays, 60 stations, same seeds for every policy
(`results/eval_summary.csv`, `results/eval_comparison.png`).

| Policy | Return | Service rate | Lost trips/day | Truck km/day | Failed docks/day |
|---|---|---|---|---|---|
| No rebalancing | −424.4 | 86.9% | 378 | 0 | 185 |
| Threshold (default 0.2/0.8/0.5) | −327.3 | 90.3% | 283 | 124 | 79 |
| Threshold (tuned 0.1/0.8/0.4) | **−311.8** | **91.0%** | **263** | 137 | 85 |
| PPO (750k-step checkpoint) | −320.0 | 90.6% | 273 | **86** | 119 |

- MaskablePPO, 1.5M steps (~55 min on 2 CPU cores); learning curve in `results/training_curve.png`.
- The checkpoint was selected by return on 20 validation days drawn from the **training** demand
  (750k: −285.1 vs −304.9 at 1.5M); October data was not used for selection.
- PPO beats no rebalancing and the default threshold, and matches the tuned threshold's
  service within 0.4 pts while driving **38% fewer km** — but it does not yet beat the tuned
  threshold on return. It lets more riders find full docks (119 vs 85 failed docks/day).
- Trained agents: `models/ppo_rebike_best.zip` (selected) and `models/ppo_rebike.zip` (final, 1.5M).

## Status

- [x] Data pipeline and demand model (real dock counts, train/test months)
- [x] Simulator environment + tests (warm start, action masking)
- [x] Baselines (no rebalancing, default + tuned threshold)
- [x] First full PPO training run and held-out evaluation
- [ ] PPO tuning to beat the tuned threshold (reward weights, entropy/LR schedule, longer training)
- [ ] Ablations, final graphs and 2-page report
