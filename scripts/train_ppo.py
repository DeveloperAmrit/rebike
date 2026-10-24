"""Train a Maskable PPO rebalancing agent.

    python scripts/train_ppo.py --config configs/default.yaml --subproc
    python scripts/train_ppo.py --timesteps 20000      # quick smoke run

Infeasible (station, transfer) actions are masked out via `env.action_masks()`
(sb3-contrib MaskablePPO). Episode statistics are logged to
results/train_monitor.csv for the learning-curve plot.
"""
import argparse
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from rebike.config import load_config
from rebike.data import DemandModel
from rebike.env import make_env

INFO_KEYS = ("served", "lost", "km", "service_rate")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--name", default="ppo_rebike")
    p.add_argument("--subproc", action="store_true", help="run envs in subprocesses")
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

    model = MaskablePPO(
        "MlpPolicy", venv,
        n_steps=pc["n_steps"], batch_size=pc["batch_size"], n_epochs=pc["n_epochs"],
        learning_rate=pc["learning_rate"], gamma=pc["gamma"], gae_lambda=pc["gae_lambda"],
        clip_range=pc["clip_range"], ent_coef=pc["ent_coef"],
        policy_kwargs=dict(net_arch=pc["net_arch"], activation_fn=torch.nn.Tanh),
        tensorboard_log=pc.get("log_dir"), seed=pc["seed"], verbose=1,
    )
    ckpt = CheckpointCallback(save_freq=max(250_000 // pc["n_envs"], 1),
                              save_path=str(model_dir / "checkpoints"), name_prefix=args.name)
    model.learn(total_timesteps=args.timesteps or pc["total_timesteps"], callback=ckpt, tb_log_name=args.name)

    model.save(model_dir / args.name)
    venv.save(str(model_dir / f"{args.name}_vecnorm.pkl"))
    print(f"[train] saved {model_dir / args.name}.zip")


if __name__ == "__main__":
    main()
