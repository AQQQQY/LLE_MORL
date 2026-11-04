"""Run the LLE-MORL stages, or evaluate an existing completed experiment."""
import json
import os
import time

from utils.configs import configure_args
from utils.runs import configuration_mismatches, resolve_run_dir, run_config
from utils.training import allocation_summary, split_budget


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def main(argv=None):
    args = configure_args(argv)
    config = run_config(args)
    run_dir = resolve_run_dir(args)
    plan = allocation_summary(args)
    print(f'Random seed: {args.seed}')
    print(f'Run directory: {run_dir}')
    print(json.dumps(plan, indent=2))
    if args.base_weights is not None:
        print(f'Base preference weights: {args.base_weights}')
    if args.dry_run:
        return
    os.environ['MPLBACKEND'] = 'Agg'
    if args.gpu != 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from algorithms.base_training import train_policy_scalar
    from algorithms.directional_retrain import directional_retrain
    from algorithms.fine_tune import finetune_model
    from algorithms.lle import generate_candidates, save_selected_policies, nondominated_indices
    from utils.common import generate_preference_weights, set_seed_everywhere, two_direction_shift, vec_tag
    from utils.evaluation import evaluate_policy
    from utils.metrics import evaluate_and_save
    from utils.plotting import plot_combined_pareto_front

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    config_path, manifest_path = run_dir / 'config.json', run_dir / 'run_manifest.json'
    # Prevent silently reusing checkpoints from a different seed or allocation.
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        mismatches = configuration_mismatches(previous, config, test_only=args.test_only)
        if mismatches:
            raise ValueError(f"Run configuration differs for {', '.join(mismatches)}. Use a new --exp_name.")
    elif args.test_only:
        raise FileNotFoundError(f'No release-format run at {run_dir}. Use --test_only with a completed run.')
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f'{run_dir} contains an older experiment without config.json. Use a new --exp_name.')
    run_dir.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        write_json(config_path, config)
    dirs = {name: run_dir / folder for name, folder in (
        ('base', 'base_policies'), ('directional', 'directional_retrain'), ('lle', 'lle'), ('finetune', 'finetune'))}
    for directory in dirs.values():
        directory.mkdir(exist_ok=True)
    set_seed_everywhere(args.seed)
    args.training_actual = {'base': 0, 'directional': 0, 'finetune': 0}
    reused = dict(args.training_actual)
    started = time.time()
    timings = {}
    base_weights = (np.asarray(args.base_weights, dtype=np.float32)
                    if args.base_weights is not None else
                    generate_preference_weights(args.objective_dim, args.num_base_policies, seed=args.seed))
    retrain_weights = two_direction_shift(base_weights, args.retrain_shift)
    # Three-decimal checkpoint tags must identify distinct policies/directions.
    if len({vec_tag(w) for w in base_weights}) != len(base_weights) or any(
        len({vec_tag(w) for w in [base, *directions]}) != len(directions) + 1
        for base, directions in zip(base_weights, retrain_weights)
    ):
        raise ValueError('Preference weights collide in checkpoint filenames; use distinct base preferences and a retrain shift that produces distinct directions.')
    policy_kwargs = {'net_arch': args.PPOstructure_custom}

    if args.test_only:
        if not manifest_path.exists():
            raise FileNotFoundError('No completed run manifest found; finish training first.')
        manifest = json.loads(manifest_path.read_text())
        selected = manifest['lle_policies']
        fine_entries = manifest['finetuned_policies']
        # Evaluate saved policies without regenerating or retraining candidates.
        for entries, directory in ((selected, dirs['lle']), (fine_entries, dirs['finetune'])):
            for entry in entries:
                _, reward = evaluate_policy(str(directory), entry['name'], np.asarray(entry['weight']), args,
                                             episodes=args.num_test_episodes, save_dir=None, device=device)
                entry['return'] = reward.tolist()
        result_dir = run_dir / 'evaluation'
    else:
        tick = time.time()
        for weight, steps in zip(base_weights, plan['base_steps']):
            args.base_timesteps = steps
            if (dirs['base'] / f'policy_{vec_tag(weight)}.zip').exists():
                reused['base'] += steps
            train_policy_scalar(args, str(dirs['base']), weight, policy_kwargs, device)
        timings['base'] = time.time() - tick
        tick = time.time()
        steps_iter = iter(plan['directional_steps'])
        for weight, directions in zip(base_weights, retrain_weights):
            for direction in directions:
                steps = next(steps_iter)
                name = f'policy_{vec_tag(weight)}_retrained_to_{vec_tag(direction)}.zip'
                if (dirs['directional'] / name).exists():
                    reused['directional'] += steps
                directional_retrain(args, str(dirs['base']), str(dirs['directional']), weight, direction,
                                    policy_kwargs, device, retrain_timestep=steps,
                                    evaluate_episode=args.num_test_episodes, plot=False, save_model_distance=False)
        timings['directional'] = time.time() - tick
        tick = time.time()
        selection_path = run_dir / 'selection_manifest.json'
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
            selected = selection['lle_policies']
            fine_plan = selection['finetune_plan']
            for entry in selected:
                if not (dirs['lle'] / (entry['name'] + '.zip')).exists():
                    raise FileNotFoundError(f"Selected checkpoint missing: {entry['name']}; restore it or use a new run.")
        else:
            candidates = generate_candidates(args, dirs['base'], dirs['directional'], base_weights, retrain_weights, device, dirs['lle'])
            selected = save_selected_policies(candidates, dirs['base'], dirs['directional'], base_weights,
                                             retrain_weights, device, dirs['lle'])
            fine_plan = []
            fine_selected = selected
            if not args.disable_finetune and selected:
                if args.total_timesteps is not None:
                    budget = plan['stage_budgets']['finetune']
                    count = min(len(selected), budget // 2)
                    if count < len(selected):
                        fine_selected = [selected[i] for i in np.linspace(0, len(selected) - 1, count, dtype=int)]
                        print(f'Fine-tuning budget supports {count}/{len(selected)} candidates at >=2 steps each.')
                    fine_steps = split_budget(budget, count) if count else []
                else:
                    fine_steps = [args.finetune_timesteps] * len(selected)
                fine_plan = [{**entry, 'training_steps': steps} for entry, steps in zip(fine_selected, fine_steps)]
            # Freeze selection/allocation before the first fine-tuning update.
            write_json(selection_path, {'lle_policies': selected, 'finetune_plan': fine_plan})
        timings['lle'] = time.time() - tick
        tick = time.time()
        fine_entries = []
        for entry in fine_plan:
            name, steps = entry['name'], entry['training_steps']
            path = dirs['finetune'] / f'{name}.zip'
            if path.exists():
                reused['finetune'] += steps
            else:
                candidate = PPO.load(dirs['lle'] / f'{name}.zip', device=device)
                refined = finetune_model(args, candidate, np.asarray(entry['weight']), steps, policy_kwargs, device)
                refined.save(path)
            _, reward = evaluate_policy(str(dirs['finetune']), name, np.asarray(entry['weight']), args,
                                         episodes=args.num_test_episodes, save_dir=None, device=device)
            fine_entries.append({**entry, 'return': reward.tolist()})
        timings['finetune'] = time.time() - tick
        manifest = {
            'config': config, 'allocation': plan, 'lle_policies': selected, 'finetuned_policies': fine_entries,
            'training_steps_new': args.training_actual, 'training_steps_reused': reused,
            'training_steps_total': sum(args.training_actual.values()) + sum(reused.values()),
            'evaluation_in_training_budget': False,
        }
        if args.total_timesteps is not None and manifest['training_steps_total'] > args.total_timesteps:
            raise RuntimeError('Training exceeded the global budget.')
        manifest['timings_seconds'] = {**timings, 'total': time.time() - started}
        write_json(manifest_path, manifest)
        result_dir = run_dir

    # Fine-tuned output is the nondominated union of original and refined policies.
    saved = {}
    stages = [('lle', selected)]
    if fine_entries:
        stages.append(('finetune', selected + fine_entries))
    for stage, entries in stages:
        indices = nondominated_indices(np.asarray([e['return'] for e in entries]))
        entries = [entries[i] for i in indices]
        points = [np.asarray([e['return'] for e in entries if e['base_index'] == i]).reshape(-1, args.objective_dim)
                  for i in range(len(base_weights))]
        weights = [np.asarray([e['weight'] for e in entries if e['base_index'] == i]).reshape(-1, args.objective_dim)
                   for i in range(len(base_weights))]
        destination = result_dir / stage
        destination.mkdir(parents=True, exist_ok=True)
        plot_combined_pareto_front(args, points, base_weights, str(destination), mode=stage)
        evaluate_and_save(points, weights, args, str(destination))
        for i, (returns, preferences) in enumerate(zip(points, weights)):
            prefix = 'nondominated' if stage == 'lle' else 'finetuned'
            saved[f'{prefix}_{i}'] = returns
            saved[f'{prefix}_weights_{i}'] = preferences
    np.savez(result_dir / 'pareto_front.npz', **saved)
    if args.test_only:
        write_json(result_dir / 'evaluation_manifest.json', {
            'num_test_episodes': args.num_test_episodes, 'seed': args.seed,
            'lle_policies': selected, 'finetuned_policies': fine_entries,
            'training_steps_new': args.training_actual,
        })
    print(f'Results: {result_dir}')
    print(f'New training interactions: {args.training_actual}')


if __name__ == '__main__':
    main()
