"""Deterministic run folders and compatibility with earlier experiment layouts."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from utils.training import DEFAULT_STAGE_RATIOS, normalize_stage_ratios


_EXECUTION_FIELDS = {
    'test_only', 'dry_run', 'device', 'gpu', 'output_dir', 'training_actual',
}


def _normalized_config(config):
    """Compare effective settings, not unused flags or execution location."""
    result = deepcopy({key: value for key, value in config.items()
                       if key not in _EXECUTION_FIELDS})
    # Omitted custom weights must keep existing generated-weight runs compatible.
    if result.get('base_weights') is None:
        result.pop('base_weights', None)
    disabled = result.get('disable_finetune', False)
    if result.get('total_timesteps') is not None:
        ratios = result.get('stage_ratios')
        result['stage_ratios'] = list(normalize_stage_ratios(
            DEFAULT_STAGE_RATIOS if ratios is None else ratios,
        ))
        result['base_timesteps'] = None
        result['dir_timesteps'] = None
        result['finetune_timesteps'] = 0 if disabled else None
    else:
        result.pop('stage_ratios', None)
        if disabled:
            result['finetune_timesteps'] = 0
    return result


def run_config(args):
    """Return the portable configuration saved alongside a run's checkpoints."""
    return _normalized_config(vars(args))


def configuration_mismatches(previous, current, test_only=False):
    """List changed settings; evaluation may use a different episode count."""
    previous = _normalized_config(previous)
    current = _normalized_config(current)
    ignored = {'num_test_episodes'} if test_only else set()
    return [key for key in sorted(previous.keys() | current.keys())
            if key not in ignored and previous.get(key) != current.get(key)]


def generated_run_name(args):
    """Readable budget/seed prefix plus a digest of all effective run settings.

    Evaluation episodes participate because they affect candidate selection.
    Operational flags and experiment-group names do not change run identity.
    """
    config = run_config(args)
    identity = {key: value for key, value in config.items() if key != 'exp_name'}
    # Keep existing default-run paths usable after introducing the ratio flag.
    if identity.get('stage_ratios') == list(DEFAULT_STAGE_RATIOS):
        identity.pop('stage_ratios')
    serialized = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False)
    digest = hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:12]
    prefix = f"{config['objective_dim']}d_K{config['num_base_policies']}"
    if config.get('total_timesteps') is not None:
        budget = f"total{config['total_timesteps']}"
    else:
        budget = (f"base{config['base_timesteps']}_dir{config['dir_timesteps']}"
                  f"_fine{config['finetune_timesteps']}")
    return f"{prefix}_{budget}_seed{config['seed']}_{digest}"


def _read_config(path):
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Cannot read saved run configuration: {path}') from exc
    if not isinstance(config, dict):
        raise ValueError(f'Saved run configuration must be a JSON object: {path}')
    return config


def resolve_run_dir(args):
    """Choose a generated or compatible legacy run without creating files.

    Existing exact run names take precedence. Earlier releases stored config.json
    directly in the experiment group; matching runs remain accessible there.
    Evaluation with a changed episode count may resolve a uniquely matching run.
    """
    group = Path(args.output_dir) / args.env / args.exp_name
    desired = group / generated_run_name(args)
    if desired.exists():
        return desired
    current = run_config(args)
    test_only = bool(args.test_only)
    legacy_config = group / 'config.json'
    legacy_evaluation_match = False
    if legacy_config.is_file():
        previous = _read_config(legacy_config)
        if not configuration_mismatches(previous, current):
            return group
        legacy_evaluation_match = test_only and not configuration_mismatches(
            previous, current, test_only=True,
        )
    if test_only:
        matches = [group] if legacy_evaluation_match else []
        matches.extend(path.parent for path in sorted(group.glob('*/config.json'))
                       if not configuration_mismatches(_read_config(path), current, test_only=True))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f'Multiple matching runs exist under {group}. Repeat the original '
                '--num_test_episodes value to select the intended run.'
            )
    return desired
