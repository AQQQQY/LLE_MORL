"""Pareto-front metrics for maximized objective returns."""

import os
import os.path as osp
from typing import List, Tuple

import numpy as np
from pymoo.indicators.hv import HV

from utils.common import choose_nondominated_points


def hypervolume(points: np.ndarray, ref_point: np.ndarray) -> float:
    """Compute maximization HV by negating points and reference for pymoo."""
    pts = np.asarray(points, dtype=float)
    ref = np.asarray(ref_point, dtype=float).reshape(-1)
    if pts.size == 0:
        return 0.0
    return float(HV(ref_point=-ref)(-pts.reshape(-1, len(ref))))


def sparsity(points: np.ndarray) -> float:
    """Sum squared adjacent gaps per objective, averaged over n - 1."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.size == 0 or len(pts) < 2:
        return 0.0
    return float(np.square(np.diff(np.sort(pts, axis=0), axis=0)).sum() / (len(pts) - 1))


def expected_utility(points: np.ndarray, weights_set: np.ndarray) -> float:
    """Average the best scalarized return over fixed evaluation preferences."""
    pts = np.asarray(points, dtype=float)
    wts = np.asarray(weights_set, dtype=float)
    if pts.size == 0 or wts.size == 0:
        return float('nan')  # No policy is available to evaluate.
    pts = np.atleast_2d(pts)
    wts = wts.reshape(-1, pts.shape[-1])
    return float(np.max(pts @ wts.T, axis=0).mean())


def evaluation_weights(objective_dim: int) -> np.ndarray:
    """Use the same preference distribution for every run and training stage."""
    if objective_dim == 2:
        first = np.linspace(0.0, 1.0, 101)
        return np.column_stack((first, 1.0 - first))
    return np.random.default_rng(0).dirichlet(np.ones(objective_dim), size=1000)


def default_ref_point(points: np.ndarray, pad_ratio: float = 0.1) -> np.ndarray:
    """Observed-front fallback; do not compare HV using different references."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.zeros(pts.shape[-1], dtype=float)
    mins = pts.min(axis=0)
    span = np.maximum(pts.max(axis=0) - mins, 1e-6)
    return mins - pad_ratio * span


def env_ref_point(env: str, objective_dim: int) -> np.ndarray | None:
    """Default hypervolume reference points for supported environments."""
    if env in ('Hopper', 'Ant') and objective_dim in (2, 3):
        return np.full(objective_dim, -100.0)
    if objective_dim == 2 and env == 'Swimmer':
        return np.array([-100.0, -2000.0])
    if objective_dim == 2 and env == 'Humanoid':
        return np.array([-100.0, -1000.0])
    return None


def _combine_fronts(points_list, weights_list, objective_dim):
    """Keep each set paired with its own weights; omit incomplete metadata."""
    points_list = list(points_list)
    weights_list = list(weights_list) if weights_list is not None else []
    if weights_list and len(points_list) != len(weights_list):
        raise ValueError('Each point set must have a corresponding weight set or None.')
    valid_points, valid_weights = [], []
    complete_weights = bool(weights_list)
    for index, point_set in enumerate(points_list):
        weight_set = weights_list[index] if weights_list else None
        points = np.asarray(point_set, dtype=float).reshape(-1, objective_dim) if point_set is not None else np.empty((0, objective_dim))
        weights = np.asarray(weight_set, dtype=float).reshape(-1, objective_dim) if weight_set is not None else np.empty((0, objective_dim))
        if len(weights) and len(weights) != len(points):
            raise ValueError('The number of points and weights must be the same.')
        if not len(points):
            continue
        if not np.isfinite(points).all() or not np.isfinite(weights).all():
            raise ValueError('Points and weights must contain only finite values.')
        valid_points.append(points)
        if len(weights):
            valid_weights.append(weights)
        else:
            complete_weights = False
    points = np.vstack(valid_points) if valid_points else np.empty((0, objective_dim))
    weights = np.vstack(valid_weights) if complete_weights and valid_weights else None
    # A Pareto front is a set: duplicated returns must not inflate its density.
    _, indices = np.unique(points, axis=0, return_index=True)
    indices.sort()
    points = points[indices]
    if weights is not None:
        weights = weights[indices]
    return choose_nondominated_points(points, n_obj=objective_dim, weights=weights)


def evaluate_front(points_list: List[np.ndarray], weights_list: List[np.ndarray], *, env: str, objective_dim: int) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Combine fronts, select nondominated policies, and compute HV/SP/EU."""
    front, front_weights = _combine_fronts(points_list, weights_list, objective_dim)
    ref = env_ref_point(env, objective_dim)
    reference_source = 'fixed benchmark reference'
    if ref is None:
        ref = default_ref_point(front)
        reference_source = 'observed-front fallback; not comparable across different references'
    utility_weights = evaluation_weights(objective_dim)
    metrics = {
        'hypervolume': hypervolume(front, ref),
        'sparsity': sparsity(front),
        'expected_utility': expected_utility(front, utility_weights),
        'reference_point': ref.tolist(),
        'reference_source': reference_source,
        'num_points': len(front),
        'num_evaluation_weights': len(utility_weights),
    }
    return metrics, front, front_weights


def evaluate_and_save(points_list: List[np.ndarray], weights_list: List[np.ndarray], args, save_dir: str, filename: str = 'evaluation.txt') -> dict:
    """Save metrics, fronts, policy weights, and fixed evaluation preferences."""
    metrics, front, front_weights = evaluate_front(points_list, weights_list, env=args.env, objective_dim=args.objective_dim)
    os.makedirs(save_dir, exist_ok=True)
    np.save(osp.join(save_dir, 'pareto_front.npy'), front)
    np.savetxt(osp.join(save_dir, 'pareto_front.txt'), front, delimiter=',', fmt='%.6f')
    if front_weights is not None:
        np.save(osp.join(save_dir, 'pareto_weights.npy'), front_weights)
        np.savetxt(osp.join(save_dir, 'pareto_weights.txt'), front_weights, delimiter=',', fmt='%.6f')
    np.save(osp.join(save_dir, 'evaluation_weights.npy'), evaluation_weights(args.objective_dim))

    with np.printoptions(threshold=np.inf, linewidth=200, precision=6, suppress=True):
        front_str = 'np.array(' + np.array2string(front, separator=', ') + ')\n'
        weights_str = 'np.array(' + np.array2string(front_weights, separator=', ') + ')\n' if front_weights is not None else None
    with open(osp.join(save_dir, 'pareto_front_array.txt'), 'w') as output:
        output.write(front_str)
    if weights_str is not None:
        with open(osp.join(save_dir, 'pareto_weights_array.txt'), 'w') as output:
            output.write(weights_str)
    with open(osp.join(save_dir, filename), 'w') as output:
        output.write(f"Hypervolume: {metrics['hypervolume']}\n")
        output.write(f"Sparsity: {metrics['sparsity']}\n")
        output.write(f"Expected utility: {metrics['expected_utility']}\n")
        output.write(f"Reference point: {metrics['reference_point']}\n")
        output.write(f"Reference source: {metrics['reference_source']}\n")
        output.write(f"Num points: {metrics['num_points']}\n")
        output.write(f"Evaluation weights: evaluation_weights.npy ({metrics['num_evaluation_weights']})\n")
    return metrics
