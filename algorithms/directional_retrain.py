"""Continue a base policy under a nearby preference and compare the policies."""

import os
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import mo_gymnasium as mo_gym

from utils.common import (
    make_mo_env, vec_tag, CustomTrainingCallback,
    model_distance_hungarian, get_layer_params,
)
from utils.evaluation import evaluate_policy
from utils.training import learn_with_budget


def directional_retrain(
    args, load_dir: str, save_dir: str, base_weights: np.ndarray,
    retrain_weights: np.ndarray, policy_kwargs: dict, DEVICE,
    retrain_timestep=100000, lr=None, evaluate_episode=10,
    save_model_distance=True, plot=True,
):
    """Train one direction, or load its checkpoint without additional training."""
    base_weights_tag = vec_tag(base_weights)
    retrain_weights_tag = vec_tag(retrain_weights)
    model_path = osp.join(load_dir, f"policy_{base_weights_tag}.zip")
    model_name = f"policy_{base_weights_tag}_retrained_to_{retrain_weights_tag}"
    model_new_path = osp.join(save_dir, model_name + ".zip")
    if not osp.exists(model_path):
        raise FileNotFoundError(f"Base policy not found: {model_path}")
    if args.test_only and not osp.exists(model_new_path):
        raise FileNotFoundError(f"Directional policy required by --test_only: {model_new_path}")
    os.makedirs(save_dir, exist_ok=True)

    original = PPO.load(model_path, device=DEVICE)
    model_original_weights = original.policy.state_dict()
    rewards_ori, disc_rewards_ori = evaluate_policy(
        load_dir, f"policy_{base_weights_tag}", base_weights, args,
        episodes=evaluate_episode, save_dir=None, device=DEVICE,
    )

    if osp.exists(model_new_path):
        print(f"Loading existing directional policy: {model_new_path}")
        model = PPO.load(model_new_path, device=DEVICE)
    else:
        env = mo_gym.wrappers.LinearReward(make_mo_env(args), weight=retrain_weights)
        try:
            model = PPO.load(model_path, env=env, device=DEVICE)
            model.learning_rate = args.learning_rate if lr is None else lr
            model.tensorboard_log = osp.join(save_dir, "ppo_tensorboard")
            learn_with_budget(
                model, retrain_timestep, args, "directional",
                learning_rate=lr,
                callback=CustomTrainingCallback(args.objective_dim),
                progress_bar=True, tb_log_name=f"PPO_{model_name}",
            )
            model.save(model_new_path)
        finally:
            env.close()
    model_new_weights = model.policy.state_dict()
    rewards_new, disc_rewards_new = evaluate_policy(
        save_dir, model_name, retrain_weights, args,
        episodes=evaluate_episode, save_dir=None, device=DEVICE,
    )

    # An independently trained target-weight policy is optional; it is used
    # only for diagnostic comparisons.
    model_aim_path = osp.join(load_dir, f"policy_{retrain_weights_tag}.zip")
    model_aim_weights = None
    rewards_aim = disc_rewards_aim = None
    if osp.exists(model_aim_path):
        aimed = PPO.load(model_aim_path, device=DEVICE)
        model_aim_weights = aimed.policy.state_dict()
        rewards_aim, disc_rewards_aim = evaluate_policy(
            load_dir, f"policy_{retrain_weights_tag}", retrain_weights, args,
            episodes=evaluate_episode, save_dir=None, device=DEVICE,
        )

    if save_model_distance:
        # Include every hidden layer in both networks, including custom depths.
        layer_keys = [
            key for key in model_original_weights
            if key.startswith("mlp_extractor.") and key.endswith(".weight")
        ]
        def hidden_layers(state):
            return [get_layer_params(state, key, key[:-6] + "bias") for key in layer_keys]

        ori_layers = hidden_layers(model_original_weights)
        new_layers = hidden_layers(model_new_weights)
        distance_ori_new_hungarian, _ = model_distance_hungarian(ori_layers, new_layers)
        with open(osp.join(save_dir, model_name + "_parameters_change.txt"), "w") as stream:
            for key in model_original_weights:
                stream.write(f"{key}_ori: {model_original_weights[key]}\n")
                stream.write(f"{key}_new: {model_new_weights[key]}\n")
                stream.write(f"{key}_change: {model_new_weights[key] - model_original_weights[key]}\n")
                if model_aim_weights is not None:
                    stream.write(f"{key}_aim: {model_aim_weights[key]}\n")
        with open(osp.join(save_dir, model_name + "_distance.txt"), "w") as stream:
            stream.write(f"Distance between original and new models (Hungarian): {distance_ori_new_hungarian}\n")
            if model_aim_weights is not None:
                aim_layers = hidden_layers(model_aim_weights)
                distance_ori_aim, _ = model_distance_hungarian(ori_layers, aim_layers)
                distance_new_aim, _ = model_distance_hungarian(new_layers, aim_layers)
                stream.write(f"Distance between original and aimed models (Hungarian): {distance_ori_aim}\n")
                stream.write(f"Distance between new and aimed models (Hungarian): {distance_new_aim}\n")

    if plot:
        # draw original and new rewards for comparison
        color_orange=(233/255, 133/255, 50/255,1)
        color_green=(78/255, 167/255, 46/255,1)
        color_blue=(15/255, 158/255, 213/255,1)
        objective_dim = args.objective_dim
        disc_rewards_ori = np.asarray(disc_rewards_ori).reshape(-1, objective_dim)
        disc_rewards_new = np.asarray(disc_rewards_new).reshape(-1, objective_dim)
        if model_aim_weights is not None:
            disc_rewards_aim = np.asarray(disc_rewards_aim).reshape(-1, objective_dim)
        if objective_dim == 2 or objective_dim >= 4:
            plt.figure(figsize=(6, 6))
            plt.scatter(disc_rewards_ori[:, 0], disc_rewards_ori[:, 1], label="$\\theta_{\\omega_1}$",color=color_blue, s=200)
            if model_aim_weights is not None:
                plt.scatter(disc_rewards_aim[:, 0], disc_rewards_aim[:, 1], label="$\\theta_{\\omega_2}$",color=color_orange, s=200)
            plt.scatter(disc_rewards_new[:, 0], disc_rewards_new[:, 1], label="$\\theta_{\\omega'}$",color=color_green, s=200)
            if objective_dim >= 4:
                print(f"[warn] D={objective_dim}: plot shows R0 vs R1 only.")
            plt.xlabel('Reward 1',fontsize=15)
            plt.ylabel('Reward 2',fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.locator_params(axis='y', nbins=6, tight=True)
            plt.locator_params(axis='x', nbins=6, tight=True)
            # plt.title('Rewards of the original and retrained policies')
            plt.legend(fontsize=20)
            plt.tight_layout(rect=[0, 0, 1, 1])  # Prevents labels from being cut off
            plt.savefig(osp.join(save_dir, f'policy_{base_weights_tag}_retrained_to_{retrain_weights_tag}_rewards.png'))
            plt.close()
        elif objective_dim == 3:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(disc_rewards_ori[:, 0], disc_rewards_ori[:, 1], disc_rewards_ori[:, 2], label="$\\theta_{\\omega_1}$",color=color_blue, s=200)
            if model_aim_weights is not None:
                ax.scatter(disc_rewards_aim[:, 0], disc_rewards_aim[:, 1], disc_rewards_aim[:, 2], label="$\\theta_{\\omega_2}$",color=color_orange, s=200)
            ax.scatter(disc_rewards_new[:, 0], disc_rewards_new[:, 1], disc_rewards_new[:, 2], label="$\\theta_{\\omega'}$",color=color_green, s=200)
            ax.set_xlabel('Reward 1',fontsize=15)
            ax.set_ylabel('Reward 2',fontsize=15)
            ax.set_zlabel('Reward 3',fontsize=15)
            ax.tick_params(labelsize=15)
            plt.legend(fontsize=20)
            plt.tight_layout(rect=[0, 0, 1, 1])
            plt.savefig(osp.join(save_dir, f'policy_{base_weights_tag}_retrained_to_{retrain_weights_tag}_rewards.png'))
            plt.close()
        

        # draw model parameter heatmap (used for paper's figure)
        plt.figure(figsize=(15, 8))
        draw_keys=['mlp_extractor.policy_net.0.weight','mlp_extractor.value_net.0.weight']
        name_keys=['policy_net','value_net']
        ori = [model_original_weights[k].cpu().detach().numpy() for k in draw_keys]
        new = [model_new_weights[k].cpu().detach().numpy() for k in draw_keys]
        if model_aim_weights is not None:
            aim = [model_aim_weights[k].cpu().detach().numpy() for k in draw_keys]
        cmap_name="RdYlGn"
        # cmap_name="coolwarm"
        for i in range(len(draw_keys)):
            # for each key, draw the heatmap of the parameters of original moedel / new model / aimed model / change
            plt.subplot(2,3,i*3+1)
            plt.imshow(ori[i], cmap=cmap_name, aspect='auto')
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.subplots_adjust(wspace=0.4, hspace=0.2)
            # plt.title(f'{name_keys[i]} trained with {loaded_policy_scalar}',fontsize=8)
            cbar=plt.colorbar()
            cbar.ax.tick_params(labelsize=15)

            plt.subplot(2,3,i*3+2)
            plt.imshow(new[i], cmap=cmap_name, aspect='auto')
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.subplots_adjust(wspace=0.4, hspace=0.2)
            # plt.title(f'{name_keys[i]} retrained with {aim_policy_scalar}',fontsize=8)
            cbar=plt.colorbar()
            cbar.ax.tick_params(labelsize=15)

            if model_aim_weights is not None:
                plt.subplot(2,3,i*3+3)
                plt.imshow(aim[i], cmap=cmap_name, aspect='auto')
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.subplots_adjust(wspace=0.4, hspace=0.2)
                # plt.title(f'{name_keys[i]} change',fontsize=8)
                cbar=plt.colorbar()
                cbar.ax.tick_params(labelsize=15)
        
        plt.savefig(osp.join(save_dir, f'policy_{base_weights_tag}_retrained_to_{retrain_weights_tag}.png'))
        plt.close()

    if model_aim_weights is not None:
        return rewards_ori, disc_rewards_ori, rewards_new, disc_rewards_new, rewards_aim, disc_rewards_aim
    else:
        return rewards_ori, disc_rewards_ori, rewards_new, disc_rewards_new, None, None 
