# LLE-MORL

Code for **Local Policy Manifolds for Efficient Multi-Objective Reinforcement Learning**
(Qiyue Xia, Tianwei Wang, J. Michael Herrmann; NeurIPS 2026).

LLE-MORL efficiently reconstructs the Pareto front by tracing local policy
manifolds that connect policy parameters with multi-objective performance. Its
locally linear extrapolation stage can also be used as a training-free plug-in
module. See the [paper](https://arxiv.org/abs/2506.04022) for details.

## Requirements

Use Python 3.10. With your environment activated, run the following from this
directory to install the dependencies in [requirements.txt](requirements.txt):

```bash
python -m pip install -r requirements.txt
```

## Train

### Automatic stage allocation

Automatic stage allocation is **enabled by default**. The default split is
60% for base training, 20% for directional retraining, and 20% for fine-tuning.
Change the split with `--stage_ratios BASE DIR FINE` (default: `3 1 1`);
for example, `--stage_ratios 7 2 1` gives 70% / 20% / 10%. These relative
weights are normalized automatically and apply only to automatic allocation.
`--total_timesteps` sets the global training budget (default: 1,000,000 steps).
Use `--seed` to control the random seed. If omitted, a random seed is generated.

For more detailed control, including custom base preference weights, omit
`--total_timesteps` and `--stage_ratios` and supply per-stage budgets to select
manual mode; see [Manual stage budgets](#manual-stage-budgets) below.

For default automatic training, run the following from this directory:

```bash
python LLE_MORL.py --env Swimmer --objective_dim 2 \
  --num_base_policies 6 --seed 0 \
  --total_timesteps 1000000
```

`--exp_name` names the experiment group (default: `LLE_MORL`), so the default
group folder is `experiments/<env>/LLE_MORL/`. Each configuration automatically
gets a run subfolder named from its parameters, for example
`2d_K6_total1000000_seed<seed>_<id>`. The short suffix distinguishes other settings,
including the extension grid and network architecture. Stage folders and
policy-training logs are created inside each run.

The total training budget is shared across all policies and stages; evaluation
is excluded. Base training is divided across K base policies, and directional
retraining across K(d−1) policies, where d is the number of objectives. After
extrapolation and global nondominated selection, the fine-tuning allocation is
divided across the selected candidates. Allocation adapts to these counts and
handles integer remainders without exceeding the budget.

**K** is the number of base policies (default: 6); change it with
`--num_base_policies`. Base preference weights are generated automatically to
evenly cover the preference space.
**δ_s** controls the preference shift for directional retraining (default: 0.1);
set it with `--retrain_shift`.
The extension grid **(alpha_min, alpha_max, alpha_step)** defaults to a range of
−1.5 to 1.5 in steps of 0.05. Change it with `--alpha_min`, `--alpha_max`, and
`--alpha_step`.

Choose the environment and number of objectives with `--env` and
`--objective_dim`:

| `--env` | `--objective_dim` |
| --- | --- |
| `Swimmer` | `2` |
| `Hopper`, `Ant` | `2` or `3` |

Use `--disable_finetune` for LLE-MORL-0; alternatively, `--stage_ratios 3 1 0`
allocates the entire budget to base and directional training.

### Manual stage budgets

To control each stage manually, omit `--total_timesteps` and `--stage_ratios`
and supply at least one positive budget in steps **per policy**:

- `--base_timesteps`: per base policy (default: `100000`).
- `--dir_timesteps`: per directional retraining (default: `10000`).
- `--finetune_timesteps`: per selected candidate (default: `1000`).

Unspecified stages use these defaults. The K, δ_s, and extension-grid flags
also apply in this mode.

```bash
python LLE_MORL.py --env Swimmer --objective_dim 2 \
  --num_base_policies 6 --seed 0 \
  --base_timesteps 100000 --dir_timesteps 10000 --finetune_timesteps 1000
```

For K base policies, d objectives, and N selected candidates, total training
uses `K × base + K(d−1) × dir + N × finetune` steps.
Neither `--total_timesteps` nor `--stage_ratios` can be combined with positive
manual stage budgets.

To specify base preference weights, pass a list of base preference vectors with
`--base_weights`, as shown below.
This flag requires manual stage budgets. Each vector must have one nonnegative
value per objective and sum to 1. K is always determined from the number of
vectors, overriding any `--num_base_policies` value.

```bash
python LLE_MORL.py --env Swimmer --objective_dim 2 \
  --seed 0 \
  --base_weights '[[1,0],[0.3,0.7],[0,1]]' \
  --base_timesteps 100000 --dir_timesteps 10000 --finetune_timesteps 1000
```

Use `--disable_finetune` (or `--finetune_timesteps 0`) for LLE-MORL-0.
`--help` lists all options.

## Reference points in evaluation metrics

| Environment | Objectives | Default reference point |
| --- | --- | --- |
| Swimmer | 2 | `[-100, -2000]` |
| Hopper, Ant | 2 | `[-100, -100]` |
| Hopper, Ant | 3 | `[-100, -100, -100]` |

To change these values, edit `env_ref_point()` in
[utils/metrics.py](utils/metrics.py). Provide one value per objective.

**Note:** When calculating hypervolume, each reference-point component must be
lower than the corresponding component of every point in the evaluated Pareto
front. Check and adjust the defaults for your results, and use the same
reference point for all runs being compared.

## Citation

Please cite the [arXiv version](https://arxiv.org/abs/2506.04022) for now.
The NeurIPS citation will be updated when available.

```bibtex
@misc{xia2026local,
  title={Interpretability by Design for Efficient Multi-Objective Reinforcement Learning},
  author={Qiyue Xia and Tianwei Wang and J. Michael Herrmann},
  year={2026},
  eprint={2506.04022},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2506.04022}
}
```

Built with [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3),
[MO-Gymnasium](https://github.com/Farama-Foundation/MO-Gymnasium),
[Gymnasium](https://github.com/Farama-Foundation/Gymnasium),
[PyTorch](https://github.com/pytorch/pytorch), and
[pymoo](https://github.com/anyoptimization/pymoo).
