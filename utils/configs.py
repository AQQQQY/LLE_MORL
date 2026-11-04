"""CLI parsing and dry runs need only the standard library."""
import argparse
import ast
import json
import math
import secrets


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def timestep_count(value):
    try:
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError
        return int(number)
    except (ValueError, OverflowError):
        raise argparse.ArgumentTypeError('Expected an integer step count (e.g. 150000 or 1.5e5).')


def network_structure(value):
    try:
        result = ast.literal_eval(value)
        if not isinstance(result, dict) or set(result) != {'pi', 'vf'}:
            raise ValueError
        for layers in result.values():
            if not isinstance(layers, list) or not layers or any(type(n) is not int or n < 1 for n in layers):
                raise ValueError
        return result
    except (SyntaxError, ValueError, TypeError):
        raise argparse.ArgumentTypeError("Expected {'pi': [64, 64], 'vf': [64, 64]} with positive layer sizes.")


def base_preference_weights(value):
    """Parse explicitly supplied simplex preferences without evaluating code."""
    try:
        rows = json.loads(value)
        if not isinstance(rows, list) or not rows:
            raise ValueError
        result = []
        for row in rows:
            if not isinstance(row, list) or not row:
                raise ValueError
            if any(type(component) not in (int, float) for component in row):
                raise ValueError
            weights = [float(component) for component in row]
            if any(not math.isfinite(component) or component < 0 for component in weights):
                raise ValueError
            total = math.fsum(weights)
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError
            result.append([component / total if component else 0.0 for component in weights])
        if len({tuple(row) for row in result}) != len(result):
            raise ValueError
        return result
    except (TypeError, ValueError, OverflowError):
        raise argparse.ArgumentTypeError(
            'Expected a nonempty JSON list of distinct preference vectors with finite '
            'nonnegative numbers summing to 1, e.g. [[1, 0], [0.3, 0.7]].'
        ) from None


def configure_args(argv=None):
    p = argparse.ArgumentParser(description='LLE-MORL: local policy manifolds for multi-objective RL.')
    p.add_argument('--env', choices=['Swimmer', 'HalfCheetah', 'Walker2d', 'Hopper', 'Ant', 'Humanoid'], default='Swimmer')
    p.add_argument('--objective_dim', type=int, choices=[2, 3], default=2)
    p.add_argument('--exp_name', default='LLE_MORL', help='Experiment group folder; parameter-based run names are generated inside it.')
    p.add_argument('--output_dir', default='experiments')
    p.add_argument('--seed', type=int, help='Random seed; omitted generates a fresh seed that is printed and saved. Supply the saved seed to resume or evaluate a run.')
    p.add_argument('--num_base_policies', type=int, default=6, help='Number of generated base preferences (default: 6); overridden by the list length when --base_weights is supplied.')
    p.add_argument('--base_weights', type=base_preference_weights, metavar='JSON', help='Explicit base preference vectors as a quoted JSON list; requires manual stage budgets. K is always set to the list length.')
    p.add_argument('--num_test_episodes', type=int, default=100, help='Independent seeded episodes per evaluated policy.')
    p.add_argument('--total_timesteps', type=timestep_count, help='Global training budget (default: 1e6), automatically split by --stage_ratios; excludes evaluation. Omit when supplying manual stage budgets.')
    p.add_argument('--stage_ratios', nargs=3, metavar=('BASE', 'DIR', 'FINE'), help='Automatic allocation weights (default: 3 1 1); base and directional must be positive, fine-tuning may be zero. Accepts relative weights such as 60 20 20 or 0.6 0.2 0.2.')
    p.add_argument('--base_timesteps', type=timestep_count, help='Select manual allocation: steps per base policy (manual default: 100000).')
    p.add_argument('--dir_timesteps', type=timestep_count, help='Select manual allocation: steps per directional policy (manual default: 10000).')
    p.add_argument('--finetune_timesteps', type=timestep_count, help='Positive values select manual allocation: steps per selected candidate (manual default: 1000); <=0 only disables fine-tuning.')
    p.add_argument('--disable_finetune', action='store_true', help='LLE-MORL-0: leave the fine-tuning allocation unused.')
    p.add_argument('--retrain_shift', type=float, default=0.1)
    p.add_argument('--alpha_min', type=float, default=-1.5)
    p.add_argument('--alpha_max', type=float, default=1.5)
    p.add_argument('--alpha_step', type=float, default=0.05)
    p.add_argument('--lle3d_reward_calibrate', type=str2bool, default=False, help='Optional reward-space rescaling; off for the paper configuration.')
    p.add_argument('--preference_boundary', choices=['project', 'raw'], default='project', help='Project extrapolated fine-tuning weights onto the simplex, or retain raw affine weights.')
    p.add_argument('--ppo_n_steps', type=int, default=512)
    p.add_argument('--ppo_batch_size', type=int, default=16, help='512 / 16 = 32 minibatches per rollout (paper Table 3).')
    p.add_argument('--learning_rate', type=float, default=3e-4)
    p.add_argument('--PPOstructure_custom', type=network_structure, default={'pi': [64, 64], 'vf': [64, 64]})
    p.add_argument('--gpu', default='all', help='CUDA_VISIBLE_DEVICES value, or all (default).')
    p.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N.')
    p.add_argument('--test_only', type=str2bool, nargs='?', const=True, default=False, help='Evaluate a completed run; never train.')
    p.add_argument('--dry_run', action='store_true', help='Print allocation without RL dependencies or creating files.')
    p.add_argument('--num_envs', type=int, default=1, help='Only 1 is supported by this sequential implementation.')
    p.add_argument('--use_wandb', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--project', default='lle_morl', help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.objective_dim == 3 and a.env not in ('Hopper', 'Ant'):
        p.error('Three objectives are supported only for Hopper and Ant.')
    if a.base_weights is not None:
        if any(len(weight) != a.objective_dim for weight in a.base_weights):
            p.error(f'Each --base_weights vector must have {a.objective_dim} components for --objective_dim {a.objective_dim}.')
        a.num_base_policies = len(a.base_weights)
    else:
        if a.num_base_policies < 2:
            p.error('Use at least 2 generated base policies.')
    if a.num_test_episodes < 1:
        p.error('Use at least 1 test episode.')
    if a.seed is not None and a.seed < 0:
        p.error('--seed must be nonnegative.')
    if a.num_envs != 1:
        p.error('--num_envs must be 1; vectorized training is not implemented.')
    if a.use_wandb:
        p.error('W&B logging is not implemented; TensorBoard logs are saved locally.')
    if not math.isfinite(a.retrain_shift):
        p.error('--retrain_shift must be finite.')
    if not all(math.isfinite(v) for v in (a.alpha_min, a.alpha_max, a.alpha_step)) or a.alpha_step <= 0 or a.alpha_min > a.alpha_max:
        p.error('Use finite alpha_min <= alpha_max and alpha_step > 0.')
    intervals = (a.alpha_max - a.alpha_min) / a.alpha_step
    if not math.isclose(intervals, round(intervals), abs_tol=1e-7):
        p.error('The alpha range must be divisible by --alpha_step.')
    if a.ppo_n_steps < 2 or a.ppo_batch_size < 2:
        p.error('PPO rollout and minibatch sizes must be at least 2.')
    if not math.isfinite(a.learning_rate) or a.learning_rate <= 0:
        p.error('--learning_rate must be positive and finite.')
    if not a.exp_name or a.exp_name in ('.', '..') or '/' in a.exp_name or '\\' in a.exp_name:
        p.error('--exp_name must be a single directory name.')
    a.disable_finetune |= a.finetune_timesteps is not None and a.finetune_timesteps <= 0
    # Resolve the mode before filling defaults, so explicit per-policy budgets
    # remain distinguishable from the automatic global budget.
    manual_requested = (
        a.base_timesteps is not None
        or a.dir_timesteps is not None
        or (a.finetune_timesteps is not None and a.finetune_timesteps > 0)
    )
    if a.base_weights is not None and (
        not manual_requested or a.total_timesteps is not None or a.stage_ratios is not None
    ):
        p.error('--base_weights requires explicit manual stage budgets; omit --total_timesteps and --stage_ratios.')
    if a.stage_ratios is not None and manual_requested:
        p.error('--stage_ratios cannot be combined with manual stage budgets (except nonpositive --finetune_timesteps).')
    if a.total_timesteps is None and not manual_requested:
        a.total_timesteps = 1_000_000
    if a.total_timesteps is not None:
        if manual_requested:
            p.error('--total_timesteps cannot be combined with manual stage budgets (except --finetune_timesteps 0).')
        from utils.training import DEFAULT_STAGE_RATIOS, normalize_stage_ratios, stage_budgets, split_budget
        if a.total_timesteps <= 0:
            p.error('--total_timesteps must be positive.')
        try:
            a.stage_ratios = list(normalize_stage_ratios(
                a.stage_ratios if a.stage_ratios is not None else DEFAULT_STAGE_RATIOS))
            a.disable_finetune |= a.stage_ratios[2] == 0
            budgets = stage_budgets(a.total_timesteps, a.stage_ratios)
            split_budget(budgets['base'], a.num_base_policies)
            split_budget(budgets['directional'], a.num_base_policies * (a.objective_dim - 1))
        except ValueError as exc:
            p.error(str(exc))
    else:
        a.base_timesteps = a.base_timesteps if a.base_timesteps is not None else 100_000
        a.dir_timesteps = a.dir_timesteps if a.dir_timesteps is not None else 10_000
        a.finetune_timesteps = a.finetune_timesteps if a.finetune_timesteps is not None else 1000
        if a.base_timesteps < 2 or a.dir_timesteps < 2 or a.finetune_timesteps == 1:
            p.error('Each enabled manual training call needs at least 2 steps.')
    if a.seed is None:
        if a.test_only:
            p.error('--test_only requires --seed from the saved run configuration.')
        a.seed = secrets.randbits(32)
    return a
