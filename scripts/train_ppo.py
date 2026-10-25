"""Train a Maskable PPO rebalancing agent.

    python scripts/train_ppo.py --config configs/default.yaml --subproc
    python scripts/train_ppo.py --timesteps 20000      # quick smoke run
    python scripts/train_ppo.py --no-mask --name ppo_nomask --timesteps 1000000   # ablation

Infeasible (station, transfer) actions are masked out via `env.action_masks()`
(sb3-contrib MaskablePPO); `--no-mask` trains plain PPO on the same env instead
(infeasible transfers are then simply clipped by the environment).

Model selection: every `ppo.eval_freq` steps the policy is evaluated
deterministically on `ppo.eval_episodes` validation days drawn from the
*training* demand model (separate seeds); the best one is saved as
`models/<name>_best.zip`. The held-out test month is never used for selection.
Episode statistics go to results/<name>_monitor.monitor.csv.
"""
import argparse
import json
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from rebike.config import load_config
from rebike.data import DemandModel
from rebike.env import RebalancingEnv, make_env

INFO_KEYS = ("served", "lost", "km", "service_rate")


def linear_schedule(start: float, end_frac: float):
    """Learning rate decays linearly from `start` to `start * end_frac`."""
    def f(progress_remaining: float) -> float:
        return start * (end_frac + (1 - end_frac) * progress_remaining)
    return f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--name", default="ppo_rebike")
    p.add_argument("--subproc", action="store_true", help="run envs in subprocesses")
    p.add_argument("--no-mask", action="store_true", help="ablation: plain PPO without action masking")
    args = p.parse_args()

    cfg = load_config(args.config)
    pc = cfg["ppo"]
    model_dir = Path(pc["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(cfg["eval"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    demand = DemandModel.load(Path(cfg["env"].model_path))
    fns = [make_env(cfg["env"], demand) for _ in range(pc["n_envs"])]
    venv = (SubprocVecEnv if args.subproc else DummyVecEnv)(fns)
    venv = VecMonitor(venv, filename=str(out_dir / f"{args.name}_monitor"), info_keywords=INFO_KEYS)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=pc["gamma"])
    venv.seed(pc["seed"])

    algo = PPO if args.no_mask else MaskablePPO
    lr = pc["learning_rate"]
    if pc.get("lr_final_frac") is not None:
        lr = linear_schedule(pc["learning_rate"], pc["lr_final_frac"])
    model = algo(
        "MlpPolicy", venv,
        n_steps=pc["n_steps"], batch_size=pc["batch_size"], n_epochs=pc["n_epochs"],
        learning_rate=lr, gamma=pc["gamma"], gae_lambda=pc["gae_lambda"],
        clip_range=pc["clip_range"], ent_coef=pc["ent_coef"],
        policy_kwargs=dict(net_arch=pc["net_arch"], activation_fn=torch.nn.Tanh),
        tensorboard_log=pc.get("log_dir"), seed=pc["seed"], verbose=1,
    )

    # validation on training-demand days (seeds disjoint from training/eval seeds)
    eval_env = VecMonitor(DummyVecEnv([lambda: RebalancingEnv(cfg["env"], demand)]))
    eval_env.seed(pc.get("val_seed", 7000))
    # same wrapper type as training (required by SB3), but raw rewards and no stat updates
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)
    eval_cls = EvalCallback if args.no_mask else MaskableEvalCallback
    evaluator = eval_cls(
        eval_env, n_eval_episodes=pc.get("eval_episodes", 10),
        eval_freq=max(pc.get("eval_freq", 100_000) // pc["n_envs"], 1),
        best_model_save_path=str(model_dir / f"{args.name}_best_tmp"),
        log_path=str(out_dir / f"{args.name}_validation"), deterministic=True, verbose=1,
    )
    ckpt = CheckpointCallback(save_freq=max(250_000 // pc["n_envs"], 1),
                              save_path=str(model_dir / "checkpoints"), name_prefix=args.name)
    model.learn(total_timesteps=args.timesteps or pc["total_timesteps"],
                callback=[ckpt, evaluator], tb_log_name=args.name)

    model.save(model_dir / args.name)
    venv.save(str(model_dir / f"{args.name}_vecnorm.pkl"))
    best = model_dir / f"{args.name}_best_tmp" / "best_model.zip"
    if best.exists():
        best.replace(model_dir / f"{args.name}_best.zip")
        best.parent.rmdir()
    (out_dir / f"{args.name}_selection.json").write_text(json.dumps(dict(
        best_validation_return=float(evaluator.best_mean_reward),
        masked=not args.no_mask, extra_features=cfg["env"].extra_features,
        total_timesteps=int(model.num_timesteps)), indent=2))
    print(f"[train] saved {model_dir / args.name}.zip and {args.name}_best.zip "
          f"(best validation return {evaluator.best_mean_reward:.1f})")


if __name__ == "__main__":
    main()
