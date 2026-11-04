"""Refine a candidate using an independent optimizer and its matched preference."""

import numpy as np
import mo_gymnasium as mo_gym
from stable_baselines3 import PPO

from utils.common import CustomTrainingCallback, make_mo_env
from utils.training import learn_with_budget


def finetune_model(args, model, policy_scalar, finetune_timestep, policy_kwargs,
                   DEVICE, evaluate_episode=1):
    if args.test_only:
        raise ValueError("Fine-tuning is disabled by --test_only; load a saved policy instead.")
    policy_scalar = np.asarray(policy_scalar, dtype=np.float64)
    raw_preference = getattr(args, "preference_boundary", "project") == "raw"
    if (policy_scalar.shape != (args.objective_dim,)
            or not np.all(np.isfinite(policy_scalar))
            or not np.isclose(policy_scalar.sum(), 1.0)):
        raise ValueError("Fine-tuning requires a finite preference vector summing to one.")
    if not raw_preference and np.any(policy_scalar < 0):
        raise ValueError("Projected fine-tuning preferences must lie on the probability simplex.")

    env = mo_gym.wrappers.LinearReward(make_mo_env(args), weight=policy_scalar)
    try:
        # Each candidate starts with its own optimizer; only policy parameters
        # transfer from the extension stage.
        refined = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.ppo_n_steps,
            batch_size=args.ppo_batch_size,
            gamma=model.gamma,
            seed=args.seed,
            device=DEVICE,
            policy_kwargs=policy_kwargs,
            verbose=0,
        )
        refined.policy.load_state_dict(model.policy.state_dict())
        learn_with_budget(
            refined, finetune_timestep, args, "finetune",
            callback=CustomTrainingCallback(args.objective_dim),
            progress_bar=True,
        )
        return refined
    finally:
        env.close()
