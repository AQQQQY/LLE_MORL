import numpy as np, torch
from stable_baselines3 import PPO
from utils.common import make_mo_env,vec_tag, set_seed_everywhere
from utils.plotting import plot_single_policy, plot_combined, save_fig
import os
import os.path as osp
import matplotlib.pyplot as plt
import mo_gymnasium as mo_gym
from typing import List
from gymnasium.wrappers import RecordVideo



def evaluate_policies(exp_save_dir, scalar_weights, args, episodes=10, save_dir='plot-results', device='cpu', video=False, plot=False):
    '''
    Load the trained policy from folder and evaluate according to its scalarization weight
    The returns is list of E-D arrays or 1-D arrays (episodes' mean case) rewards, length of the list is K, where
    E is the number of episodes, D is the objective dimension.
    '''
    os.makedirs(osp.join(exp_save_dir, save_dir), exist_ok=True)
    os.makedirs(osp.join(exp_save_dir, 'videos'), exist_ok=True)

    # Lists to hold reward data for all policies
    all_returns: List[np.ndarray]       = []   # undiscounted
    all_disc_returns: List[np.ndarray]  = []   # discounted
    all_tags: List[str]                 = []

    # Set seeds for reproducibility
    set_seed_everywhere(args.seed)


    for policy_idx in range(scalar_weights.shape[0]):
        w = scalar_weights[policy_idx]
        w_tag = vec_tag(w)
        print(f"Evaluating policy_{w_tag}")
        
        # Check if the model file exists
        model_path = osp.join(exp_save_dir, f'policy_{w_tag}.zip')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Policy checkpoint not found: {model_path}")

        # # Create the multi-objective mujoco environment
        # env_mo = make_mo_env(args, render_mode='rgb_array')
        env_mo = make_mo_env(args, render_mode='rgb_array' if video else None)
        env = mo_gym.wrappers.LinearReward(env_mo, weight=w)

        # Record video for the evaluation
        if not os.path.exists(osp.join(exp_save_dir, 'videos', f'policy_{w_tag}')):
            os.makedirs(osp.join(exp_save_dir, 'videos', f'policy_{w_tag}'))
        video_path = osp.join(exp_save_dir, 'videos', f'policy_{w_tag}')
        if video:
            env = RecordVideo(env, video_folder=video_path, episode_trigger=lambda e: True)

        # Load the PPO agent
        model = PPO.load(model_path, env=env, device=device)

        # Evaluate the agent
        num_episodes = episodes

        ep_returns, ep_disc_returns = [], []

        for episode in range(num_episodes):
            obs,info = env.reset(seed=args.seed + episode)
            done = False
            total_reward_vector = np.zeros(args.objective_dim)
            total_disc_vector = np.zeros(args.objective_dim) # Initialize for this episode
            gamma = 1.0  # Represents gamma^t for current step t (starting at t=0)

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                vector_reward = info['vector_reward']
                total_reward_vector += vector_reward
                total_disc_vector += gamma * vector_reward
                gamma *= model.gamma

            ep_returns.append(total_reward_vector)     
            ep_disc_returns.append(total_disc_vector)

        env.close()

        # Store the results for this policy
        ep_means = np.mean(np.array(ep_returns), axis=0)
        ep_disc_means = np.mean(np.array(ep_disc_returns), axis=0)
        all_returns.append(ep_means)        
        all_disc_returns.append(ep_disc_means)
        all_tags.append(w_tag)


        # Plot r1-r2 figure
        if plot:
            plot_single_policy(ep_disc_means, w_tag, osp.join(exp_save_dir, save_dir), title=f"policy_{w_tag}_discounted_r ", objective_dim=args.objective_dim, fname=f"policy_{w_tag}_discounted_r.png")


    # After evaluating all policies, plot the combined discounted rewards figure
    if plot:
        plot_combined(all_disc_returns, all_tags, osp.join(exp_save_dir, save_dir), title="Combined Discounted Rewards", objective_dim=args.objective_dim, fname="combined_discounted_rewards.png") 

    
    return all_returns, all_disc_returns, all_tags


def evaluate_policy(load_dir, model_filename, scalar_weight, args, episodes=10, save_dir='plot-results', device='cpu', video=False, plot=False):
    '''
    Load one trained policy from folder and evaluate according to its scalarization weight
    The returns is a E-D arrays or 1-D arrays (episodes' mean case) rewards, where E is the number of episodes, D is the objective dimension.
    model_filename: the name of the model file to load, without '.zip' e.g., 'policy_0p5_0p5'
    save_dir could be None, in which case the results will not be saved.
    '''

    # Check if the load directory exists
    if not os.path.exists(load_dir):
        raise FileNotFoundError(f"Load directory {load_dir} does not exist.")
    # Check if the model file exists
    model_path = osp.join(load_dir, model_filename+ '.zip')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Policy checkpoint not found: {model_path}")

    # Create the save directory if it doesn't exist
    if save_dir is not None:
        os.makedirs(osp.join(load_dir, save_dir), exist_ok=True)

    # Set seeds for reproducibility
    set_seed_everywhere(args.seed)


    w = scalar_weight
    w_tag = vec_tag(w)
    print(f"Evaluating {model_filename} with weight {w_tag}")
        

    # # Create the multi-objective mujoco environment
    env_mo = make_mo_env(args, render_mode='rgb_array' if video else None)
    env = mo_gym.wrappers.LinearReward(env_mo, weight=w)

    # Record video for the evaluation
    if video:
        if not os.path.exists(osp.join(load_dir, 'videos', f'policy_{w_tag}')):
            os.makedirs(osp.join(load_dir, 'videos', f'policy_{w_tag}'))
        video_path = osp.join(load_dir, 'videos', f'policy_{w_tag}')
    
        env = RecordVideo(env, video_folder=video_path, episode_trigger=lambda e: True)

    # Load the PPO agent
    model = PPO.load(model_path, env=env, device=device)

    # Evaluate the agent
    num_episodes = episodes

    ep_returns, ep_disc_returns = [], []

    for episode in range(num_episodes):
        obs,info = env.reset(seed=args.seed + episode)
        done = False
        total_reward_vector = np.zeros(args.objective_dim)
        total_disc_vector = np.zeros(args.objective_dim) # Initialize for this episode
        gamma = 1.0  # Represents gamma^t for current step t (starting at t=0)

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            vector_reward = info['vector_reward']
            total_reward_vector += vector_reward
            total_disc_vector += gamma * vector_reward
            gamma *= model.gamma

        ep_returns.append(total_reward_vector)     
        ep_disc_returns.append(total_disc_vector)

    env.close()

    # Store the results for this policy
    ep_means = np.mean(np.array(ep_returns), axis=0)
    ep_disc_means = np.mean(np.array(ep_disc_returns), axis=0)

    # Plot r1-r2 figure
    if plot:
        plot_single_policy(ep_disc_means, w_tag, osp.join(load_dir, save_dir), title=f"{model_filename}_discounted_r ", objective_dim=args.objective_dim, fname=f"{model_filename}_discounted_r.png")

    
    return ep_means, ep_disc_means
    

def evaluate_policy_env(args, env, model, scalar_weight, DEVICE, evaluate_episode=1):
    """
    Evaluate the policy with the given scalarization weight
    """

    set_seed_everywhere(args.seed)
    num_episodes = evaluate_episode
    ep_reward_vectors = []
    ep_disc_vectors = []

    for episode in range(num_episodes):
        obs,info = env.reset(seed=args.seed + episode)
        done = False
        total_reward_vector = np.zeros(scalar_weight.shape[0])
        total_disc_vector = np.zeros(args.objective_dim) # Initialize for this episode
        gamma = 1.0  # Represents gamma^t for current step t (starting at t=0)
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            vector_reward = info['vector_reward']
            total_reward_vector += vector_reward
            total_disc_vector += gamma * vector_reward
            gamma *= model.gamma

        ep_reward_vectors.append(total_reward_vector)
        ep_disc_vectors.append(total_disc_vector)

    # Store the results for this policy
    ep_means = np.mean(np.array(ep_reward_vectors), axis=0)
    ep_disc_means = np.mean(np.array(ep_disc_vectors), axis=0)
 
    return ep_means, ep_disc_means
