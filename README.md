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
| **State** `s ∈ R^(6N+3)` | station fill levels `inv/cap` (N), one-hot truck location (N), truck load (1), `sin/cos` time of day (2), expected net inflow next hour (N), bikes in transit arriving in ≤30 min (N), truck travel time to each station (N), projected fill in 1 h (N). N = 60 → 363 dims |
| **Action** `Discrete(N·K)` | *target-fill actions*: next station × target fill level `{20, 35, 50, 65, 80%}` or "drive there, move nothing" → 60 × 6 = 360 actions. The truck moves as many bikes as needed (within its load/space) to bring the station to the target on arrival. Infeasible choices are **masked** (`env.action_masks()`, MaskablePPO) |
| **Reward** | *counterfactual*: `r = −(lost − lost_shadow) − 0.2·km − 0.25·(failed − failed_shadow)`; the shadow is the same day, same riders, without a truck (see below) |
| **Transition** | event-driven (semi-MDP): the clock advances by truck travel + handling time; rider demand is simulated in 5-minute ticks meanwhile |
| **Episode** | one weekday, 06:00–22:00 |
| **Initial state** | *warm start*: fleet at 40% of docks, one simulated day without rebalancing, then an overnight crew fixes 30% of each station's imbalance — so mornings start as unbalanced as commuting makes them |

**Reward design.** All results are reported with the *absolute* reward
`−lost − 0.2·km − 0.25·failed_docks` (a km and a redirected rider expressed as
fractions of a lost trip; total demand is exogenous, so `−lost` ranks policies like
`+served`). For training we subtract the same quantity measured on a **shadow day**:
every episode simulates the day twice with identical pre-sampled riders, once with
the truck and once without. The shadow does not depend on the agent, so the optimal
policy is unchanged, but the large day-to-day demand noise cancels out of the
return (validation std. dropped from ≈80 to ≈35). This was the change that let PPO
overtake the tuned threshold (see ablations).

**Common random numbers.** Each day's riders (Poisson counts per tick and the
random numbers that choose destinations) are drawn at reset, so on a given seed
every policy faces exactly the same riders; all comparisons are paired.

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
configs/default.yaml          all data / env / PPO / eval settings
src/rebike/data.py            download trips, build demand model (.npz)
src/rebike/env.py             Gymnasium environment (masks, target actions, shadow reward)
src/rebike/baselines.py       no-rebalancing and threshold policies
src/rebike/policies.py        loads any trained agent with its matching env, shared episode runner
scripts/prepare_data.py       data pipeline (train + test demand models)
scripts/tune_threshold.py     grid search for the threshold baseline (training demand)
scripts/train_ppo.py          Maskable PPO training, LR decay, validation callback, --no-mask ablation
scripts/select_checkpoint.py  pick the final agent on 30 validation days (training demand)
scripts/evaluate.py           held-out comparison of baselines and agents + plots
scripts/analysis.py           hourly service profile, demand-robustness sweep
scripts/plot_training.py      learning curves
scripts/make_report_tables.py LaTeX tables for the report from the result CSVs
tests/test_env.py             environment tests (conservation, masks, warm start, shadow reward, CRN)
models/ppo_v4_selected.zip    final agent (target actions + counterfactual reward, 2M steps)
models/                       earlier agents v1-v3 and the no-mask ablation
results/                      tuning, training logs, evaluation tables and figures
report/                       2-page two-column report (LaTeX)
documentation/                assignment brief, proposal, team
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
python scripts/prepare_data.py

# 2. tests
pytest -q

# 3. tune the threshold baseline on training demand -> results/threshold_tuned.json  (~5 min)
python scripts/tune_threshold.py

# 4. train the agent: 2M steps, ~50–60 min on 2 CPU cores
python scripts/train_ppo.py --subproc --name ppo_v4
python scripts/select_checkpoint.py --name ppo_v4          # -> models/ppo_v4_selected.zip

# 5. held-out comparison on 30 October days -> results/eval_summary.csv, eval_comparison.png
python scripts/evaluate.py --model ppo_v4=models/ppo_v4_selected.zip \
    --model ppo_v1=models/ppo_rebike_best.zip --model ppo_v2=models/ppo_v2_selected.zip \
    --model ppo_v3=models/ppo_v3_selected.zip --model ppo_v4_nomask=models/ppo_v4_nomask_selected.zip

# 6. analyses and figures
python scripts/analysis.py hourly     --model ppo_v4=models/ppo_v4_selected.zip
python scripts/analysis.py robustness --model ppo_v4=models/ppo_v4_selected.zip
python scripts/plot_training.py --run ppo_v4 --run ppo_v2 --run ppo_v3
python scripts/make_report_tables.py && (cd report && latexmk -pdf main.tex)

# ablation: same setup without action masking
python scripts/train_ppo.py --subproc --no-mask --name ppo_v4_nomask
python scripts/select_checkpoint.py --name ppo_v4_nomask
```

Evaluating the committed agents needs only steps 1, 3 and 5.

To use the NYC system instead, set `data.region: NYC` and the months you want in
`configs/default.yaml` (each NYC month is ~1 GB).

## Results

Held-out **October 2024** demand, 30 simulated weekdays, 60 stations, identical riders
for every policy (`results/eval_summary.csv`, `results/eval_comparison.png`).
Return = `−lost − 0.2·km − 0.25·failed_docks` per day.

| Policy | Return | Service rate | Lost trips/day | Truck km/day | Failed docks/day |
|---|---|---|---|---|---|
| No rebalancing | −413.8 | 87.3% | 366 | 0 | 190 |
| Threshold (default 0.2/0.8/0.5) | −325.3 | 90.4% | 279 | 125 | **87** |
| Threshold (tuned 0.15/0.75/0.4) | −304.0 | 91.3% | 252 | 136 | 99 |
| **PPO v4 (ours)** | **−291.5** | **91.9%** | **235** | 131 | 119 |

- Paired over the same 30 days, PPO beats the tuned threshold by **+12.5 ± 11.6 return**
  (95% CI) and saves **16.5 ± 11.5 trips/day** (better on 19/30 days), while driving 3% less
  and moving a third fewer bikes. Its weak spot is full docks.
- **Where the gain comes from** (`results/hourly_service.png`): the threshold policy is best at
  midday; PPO prepares for the evening rush and loses 25–43% fewer trips at 17–18 h.
- **Robustness** (`results/robustness.png`): with October demand scaled ×0.8 … ×1.3 (no
  retraining), PPO's advantage over the tuned threshold grows from −5 (×0.8) to +9 (×1.0) and +32 (×1.2).
- **Learning curves** (`results/training_curve.png`): v4 is still improving at 2M steps.

### Ablations (same 30 days)

| Agent | Actions | Reward | Obs | Steps | Return | Service |
|---|---|---|---|---|---|---|
| **v4 (ours)** | target fill | counterfactual | 6N+3 | 2M | **−291.5** | **91.9%** |
| v4 without action mask | target fill | counterfactual | 6N+3 | 2M | −305.7 | 91.1% |
| v3 | target fill | absolute | 6N+3 | 1.25M* | −335.3 | 89.9% |
| v2 | bike amounts | absolute | 6N+3 | 2M | −329.7 | 90.0% |
| v1 | bike amounts | absolute | 4N+3, const. LR | 1.5M | −328.4 | 90.3% |

\* stopped after plateauing. The counterfactual (shadow-day) reward is the decisive change:
without it PPO stays below even the default threshold. Action masking adds ≈14 return.

All agents were selected on validation days from the **training** demand
(`scripts/select_checkpoint.py`); October was never used for selection.

## Report

`report/main.tex` (two-column, 2 pages) — tables are generated from the result CSVs by
`scripts/make_report_tables.py`; build with `cd report && latexmk -pdf main.tex`.
A compiled draft is in `report/main.pdf`.

## Status

- [x] Data pipeline and demand model (real dock counts, train/test months)
- [x] Simulator environment + tests (warm start, masks, target actions, shadow reward, common random numbers)
- [x] Baselines (no rebalancing, default + tuned threshold)
- [x] PPO agent that beats the tuned threshold on held-out days
- [x] Ablations (reward, action design, features, masking), hourly and robustness analyses
- [x] Report draft with tables and figures
- [ ] Final report polish; optional: longer training / extra seeds
