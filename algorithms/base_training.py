"""Train one PPO base policy under a fixed preference."""

import os
import os.path as osp

import mo_gymnasium as mo_gym
from stable_baselines3 import PPO

from utils.common import CustomTrainingCallback, make_mo_env, vec_tag
from utils.training import learn_with_budget


def train_policy_scalar(args, exp_save_dir, scalar_vector, policy_kwargs, DEVICE):
    tag = vec_tag(scalar_vector)
    model_path = osp.join(exp_save_dir, f"policy_{tag}.zip")
    if osp.exists(model_path):
        print(f"Loading existing base policy: {model_path}")
        return PPO.load(model_path, device=DEVICE)
    if args.test_only:
        raise FileNotFoundError(f"Base policy required by --test_only: {model_path}")

    os.makedirs(exp_save_dir, exist_ok=True)
    env = mo_gym.wrappers.LinearReward(make_mo_env(args), weight=scalar_vector)
    try:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.ppo_n_steps,
            batch_size=args.ppo_batch_size,
            gamma=0.995,
            tensorboard_log=osp.join(exp_save_dir, "ppo_tensorboard"),
            seed=args.seed,
            device=DEVICE,
            policy_kwargs=policy_kwargs,
            verbose=1,
        )
        learn_with_budget(
            model, args.base_timesteps, args, "base",
            callback=CustomTrainingCallback(args.objective_dim),
            progress_bar=True, tb_log_name=f"PPO_{tag}",
        )
        model.save(model_path)
        return model
    finally:
        env.close()
