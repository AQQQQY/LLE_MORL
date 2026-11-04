"""Training-free affine policy extension and global candidate selection."""

import itertools
from pathlib import Path

import mo_gymnasium as mo_gym
import numpy as np
from stable_baselines3 import PPO
from tqdm import tqdm

from utils.common import make_mo_env, vec_tag
from utils.evaluation import evaluate_policy_env


def project_simplex(weights):
    """Euclidean projection onto nonnegative weights summing to one."""
    weights = np.asarray(weights, dtype=np.float64)
    sorted_weights = np.sort(weights)[::-1]
    offsets = (np.cumsum(sorted_weights) - 1) / np.arange(1, len(weights) + 1)
    rho = np.flatnonzero(sorted_weights > offsets)[-1]
    return np.maximum(weights - offsets[rho], 0)


def nondominated_indices(points):
    """Indices of unique nondominated returns, maximizing every objective."""
    points = np.asarray(points)
    if len(points) == 0:
        return np.empty(0, dtype=int)
    if not np.isfinite(points).all():
        raise ValueError('Non-finite candidate returns; inspect the run before selecting policies.')
    _, unique = np.unique(points, axis=0, return_index=True)
    active = np.zeros(len(points), dtype=bool)
    active[unique] = True
    for i in np.flatnonzero(active):
        if active[i]:
            dominated = np.all(points <= points[i], axis=1) & np.any(points < points[i], axis=1)
            active[dominated] = False
    return np.flatnonzero(active)


def coefficient_grid(args):
    intervals = round((args.alpha_max - args.alpha_min) / args.alpha_step)
    values = np.linspace(args.alpha_min, args.alpha_max, intervals + 1)
    return np.asarray(list(itertools.product(values, repeat=args.objective_dim - 1)))


def load_policy_family(base_dir, directional_dir, base_weight, retrain_weights, device, env=None):
    tag = vec_tag(base_weight)
    model = PPO.load(Path(base_dir) / f'policy_{tag}.zip', env=env, device=device)
    original = {key: value.detach().clone() for key, value in model.policy.state_dict().items()}
    deltas = []
    for weight in retrain_weights:
        direction = PPO.load(Path(directional_dir) / f'policy_{tag}_retrained_to_{vec_tag(weight)}.zip', device=device)
        deltas.append({key: value - original[key] for key, value in direction.policy.state_dict().items()})
    return model, original, deltas


def apply_extension(model, original, deltas, coefficients):
    model.policy.load_state_dict({
        key: value + sum(float(scale) * delta[key] for scale, delta in zip(coefficients, deltas))
        for key, value in original.items()
    })


def _reward_transform(args, env, model, original, deltas, weight, device):
    """Optional reward-space whitening, applied to both parameter and weight shifts."""
    transform = np.eye(len(deltas))
    if not args.lle3d_reward_calibrate or len(deltas) != 2:
        return transform
    apply_extension(model, original, deltas, [0, 0])
    _, baseline = evaluate_policy_env(args, env, model, weight, device, args.num_test_episodes)
    columns = []
    for i in range(2):
        scales = np.zeros(2)
        scales[i] = 0.1
        apply_extension(model, original, deltas, scales)
        _, reward = evaluate_policy_env(args, env, model, weight, device, args.num_test_episodes)
        columns.append(reward - baseline)
    response = np.column_stack(columns)
    norms = np.linalg.norm(response, axis=0)
    if np.min(norms) < 1e-8:
        return transform
    gram = response.T @ response
    eigenvalues, vectors = np.linalg.eigh(gram)
    if eigenvalues[0] <= 1e-8 or eigenvalues[-1] / eigenvalues[0] > 1e6:
        return np.diag(norms.mean() / norms)
    return vectors @ np.diag(1 / np.sqrt(eigenvalues)) @ vectors.T * norms.mean()


def generate_candidates(args, base_dir, directional_dir, base_weights, retrain_weights, device, save_dir):
    """Evaluate the complete inclusive grid; no PPO updates happen here.

    Store coefficients instead of all policy tensors. Selected policies are
    reconstructed once global Pareto selection has finished.
    """
    grid = coefficient_grid(args)
    returns, weights, raw_weights, coefficients, base_indices = [], [], [], [], []
    for base_index, weight in enumerate(base_weights):
        env = mo_gym.wrappers.LinearReward(make_mo_env(args), weight=weight)
        try:
            model, original, deltas = load_policy_family(base_dir, directional_dir, weight, retrain_weights[base_index], device, env)
            transform = _reward_transform(args, env, model, original, deltas, weight, device)
            for scales in tqdm(grid, desc=f'LLE base {base_index + 1}/{len(base_weights)}'):
                effective = transform @ scales
                apply_extension(model, original, deltas, effective)
                _, reward = evaluate_policy_env(args, env, model, weight, device, args.num_test_episodes)
                raw = weight + effective @ (retrain_weights[base_index] - weight)
                matched = project_simplex(raw) if args.preference_boundary == 'project' else raw
                returns.append(reward)
                raw_weights.append(raw)
                weights.append(matched)
                coefficients.append(effective)
                base_indices.append(base_index)
        finally:
            env.close()
    candidates = {
        'returns': np.asarray(returns), 'weights': np.asarray(weights),
        'raw_weights': np.asarray(raw_weights), 'coefficients': np.asarray(coefficients),
        'base_indices': np.asarray(base_indices, dtype=int),
    }
    candidates['selected_indices'] = nondominated_indices(candidates['returns'])
    np.savez(Path(save_dir) / 'candidates.npz', **candidates)
    print(f"Selected {len(candidates['selected_indices'])} globally nondominated candidates from {len(returns)}.")
    return candidates


def save_selected_policies(candidates, base_dir, directional_dir, base_weights, retrain_weights, device, save_dir):
    """Save selected LLE policies and return their portable manifest entries."""
    selected = candidates['selected_indices']
    entries = []
    for base_index, weight in enumerate(base_weights):
        indices = [int(i) for i in selected if candidates['base_indices'][i] == base_index]
        if not indices:
            continue
        model, original, deltas = load_policy_family(base_dir, directional_dir, weight, retrain_weights[base_index], device)
        for index in indices:
            apply_extension(model, original, deltas, candidates['coefficients'][index])
            name = f'candidate_{index:06d}'
            model.save(Path(save_dir) / f'{name}.zip')
            entries.append({
                'name': name, 'base_index': base_index, 'candidate_index': index,
                'weight': candidates['weights'][index].tolist(),
                'raw_weight': candidates['raw_weights'][index].tolist(),
                'coefficients': candidates['coefficients'][index].tolist(),
                'return': candidates['returns'][index].tolist(),
            })
    return entries
