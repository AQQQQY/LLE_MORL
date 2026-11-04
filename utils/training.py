"""Global budget allocation and PPO training without rollout overshoot."""
from fractions import Fraction
import math


DEFAULT_STAGE_RATIOS = (3, 1, 1)


def normalize_stage_ratios(values):
    """Express three relative stage weights as a reduced integer ratio."""
    try:
        if isinstance(values, (str, bytes)) or len(values) != 3:
            raise ValueError
        ratios = [Fraction(str(value)) for value in values]
        if ratios[0] <= 0 or ratios[1] <= 0 or ratios[2] < 0:
            raise ValueError
    except (TypeError, ValueError, ZeroDivisionError):
        raise ValueError('Stage ratios need three finite weights: base > 0, directional > 0, fine-tuning >= 0.') from None
    denominator = math.lcm(*(value.denominator for value in ratios))
    integers = [value.numerator * (denominator // value.denominator) for value in ratios]
    divisor = math.gcd(*integers)
    return tuple(value // divisor for value in integers)


def stage_budgets(total, ratios=DEFAULT_STAGE_RATIOS):
    if total < 0:
        raise ValueError('Training budget must be nonnegative.')
    base_ratio, directional_ratio, finetune_ratio = normalize_stage_ratios(ratios)
    denominator = base_ratio + directional_ratio + finetune_ratio
    base = total * base_ratio // denominator
    directional = total * directional_ratio // denominator if finetune_ratio else total - base
    return {'base': base, 'directional': directional, 'finetune': total - base - directional}


def split_budget(total, count, minimum=2):
    """Distribute integer interactions evenly, putting remainders first."""
    if count < 1 or total < count * minimum:
        raise ValueError(f'Budget {total} is too small for {count} policies (minimum {minimum} steps each).')
    quotient, remainder = divmod(total, count)
    return [quotient + (i < remainder) for i in range(count)]


def rollout_segments(total, n_steps):
    """Return (rollout size, interactions) segments without singleton rollouts."""
    if total < 2 or n_steps < 2:
        raise ValueError('PPO needs at least 2 interactions per rollout.')
    full, tail = divmod(total, n_steps)
    if tail == 1:
        full -= 1
        # A 3-step budget cannot be split into rollouts of at least 2 steps.
        tail_segments = [(3, 3)] if n_steps == 2 else [(n_steps - 1, n_steps - 1), (2, 2)]
    else:
        tail_segments = [(tail, tail)] if tail else []
    return ([(n_steps, full * n_steps)] if full else []) + tail_segments


def allocation_summary(args):
    count = args.num_base_policies
    directions = count * (args.objective_dim - 1)
    if args.total_timesteps is not None:
        ratios = normalize_stage_ratios(getattr(args, 'stage_ratios', None) or DEFAULT_STAGE_RATIOS)
        budgets = stage_budgets(args.total_timesteps, ratios)
        return {
            'mode': 'automatic', 'requested_total': args.total_timesteps,
            'stage_ratios': list(ratios),
            'stage_budgets': budgets,
            'base_steps': split_budget(budgets['base'], count),
            'directional_steps': split_budget(budgets['directional'], directions),
            'finetune_enabled': not args.disable_finetune,
            'finetune_allocation': 'Split after global nondominated selection; evaluation excluded.',
        }
    return {
        'mode': 'manual', 'requested_total': None,
        'base_steps': [args.base_timesteps] * count,
        'directional_steps': [args.dir_timesteps] * directions,
        'finetune_enabled': not args.disable_finetune,
        'finetune_steps_per_candidate': 0 if args.disable_finetune else args.finetune_timesteps,
    }


def learn_with_budget(model, timesteps, args, stage, callback=None, learning_rate=None, **kwargs):
    """Use every allocated transition, shortening final PPO rollouts as needed."""
    from stable_baselines3.common.utils import get_schedule_fn

    if args.test_only:
        raise ValueError('Training is disabled by --test_only.')
    if model.n_envs != 1:
        raise ValueError('Exact training budgets currently require one environment.')
    segments = rollout_segments(int(timesteps), args.ppo_n_steps)
    original = model.n_steps, model.batch_size, model.rollout_buffer
    initial_count = model.num_timesteps
    rate = args.learning_rate if learning_rate is None else learning_rate
    model.learning_rate = rate
    model.lr_schedule = get_schedule_fn(rate)
    if not hasattr(args, 'training_actual'):
        args.training_actual = {}
    try:
        for rollout_size, interactions in segments:
            model.n_steps = rollout_size
            model.batch_size = min(args.ppo_batch_size, rollout_size)
            model.rollout_buffer = model.rollout_buffer_class(
                rollout_size, model.observation_space, model.action_space,
                device=model.device, gamma=model.gamma, gae_lambda=model.gae_lambda,
                n_envs=model.n_envs, **model.rollout_buffer_kwargs,
            )
            before = model.num_timesteps
            model.learn(total_timesteps=interactions, callback=callback, reset_num_timesteps=False, **kwargs)
            if model.num_timesteps - before != interactions:
                raise RuntimeError('PPO stopped before completing its allocated interactions.')
    finally:
        actual = model.num_timesteps - initial_count
        args.training_actual[stage] = args.training_actual.get(stage, 0) + actual
        model.n_steps, model.batch_size, model.rollout_buffer = original
    return model
