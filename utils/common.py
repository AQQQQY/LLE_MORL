import numpy as np
import mo_gymnasium as mo_gym
from stable_baselines3.common.callbacks import BaseCallback
from scipy.optimize import linear_sum_assignment
from typing import List
import torch
import itertools, math
import hashlib



class CustomTrainingCallback(BaseCallback):
    """
    Logs *both* the scalarised return (what PPO optimises) **and**
    each component of the vector reward for any number of objectives.

    Parameters
    ----------
    objective_dim : int
        Length of the reward vector returned in ``info["reward_vector"]``.
    verbose : int
        SB3 verbosity flag (0 = silent, >0 = prints each episode).
    """

    def __init__(self, objective_dim: int, verbose: int = 0):
        super().__init__(verbose)
        self.D = objective_dim

        # history containers
        self.ep_scalarised: List[float] = []
        self.ep_vectors:     List[np.ndarray] = []

        # running containers
        self._cur_scalar: List[float]   = []
        self._cur_vec:    List[np.ndarray] = []

    # --------------------------------------------------------------------- #
    # SB3 hooks
    # --------------------------------------------------------------------- #
    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]     # scalarised
        dones   = self.locals["dones"]
        infos   = self.locals["infos"]

        for idx, done in enumerate(dones):
            self._cur_scalar.append(float(rewards[idx]))

            vec = infos[idx].get("vector_reward", np.zeros(self.D))  # shape (D,) or None
            if vec is None:
                # fallback: zeros if env doesn’t populate the field
                vec = np.zeros(self.D, dtype=np.float32)
            self._cur_vec.append(vec)

            if done:  # episode ended in *this* env copy
                ep_scalar = sum(self._cur_scalar)
                ep_vec    = np.sum(self._cur_vec, axis=0)   # shape (D,)

                # history
                self.ep_scalarised.append(ep_scalar)
                self.ep_vectors.append(ep_vec)

                # ------------ logging to SB3's logger ------------
                self.logger.record("rollout/ep_scalar_reward", ep_scalar)
                for j, val in enumerate(ep_vec):
                    self.logger.record(f"rollout/ep_reward{j}", val)

                if self.verbose > 0:
                    vec_str = ", ".join(f"{v:.2f}" for v in ep_vec)
                    print(f"[CB] ep #{len(self.ep_scalarised):3d} "
                          f"scalar={ep_scalar:.1f}  vec=[{vec_str}]")

                # reset per-episode buffers
                self._cur_scalar.clear()
                self._cur_vec.clear()

        return True



def make_mo_env(args, render_mode=None):
    """
    Create a multi-objective environment.
    """
    if args.env == "Hopper" and args.objective_dim == 2:
        env = mo_gym.make(f'mo-{args.env.lower()}-2obj-v5', render_mode=render_mode)
    elif args.env == "Ant" and args.objective_dim == 2:
        env = mo_gym.make(f'mo-{args.env.lower()}-2obj-v5', render_mode=render_mode)
    else:
        env = mo_gym.make(f'mo-{args.env.lower()}-v5', render_mode=render_mode)
    return env


def set_seed_everywhere(seed: int):
    """
    Set the random seed for reproducibility.
    """        # Set seeds for reproducibility

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_preference_weights(
    objective_dim: int,
    num_base_policies: int,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """
    Return an (N, D) array of preference vectors ω that are **evenly spread on the simplex** ∑ωᵢ = 1, ωᵢ ≥ 0.

    • For **D = 2**  →  exactly the linearly spaced vectors used in the paper:  [[1,0], [0.8,0.2], …, [0,1]].

    • For **D ≥ 3**  →  Das-Dennis simplex-lattice design:
      choose the smallest lattice parameter **H** such that C(H+D-1, D-1) ≥ N.  
      That lattice contains all integer tuples (k₁,…,k_D) with Σkᵢ = H.
      We normalise by H to get the weights.

      If the lattice contains *more* than N points
      we subsample deterministically (lexicographic order) so the function is still reproducible.

    Parameters
    ----------
    objective_dim      : int
    num_base_policies  : int
    seed               : int | None     – for the rare case you prefer random subsampling of the lattice.

    Returns
    -------
    np.ndarray   shape = (num_base_policies, objective_dim)
    """
    if objective_dim < 2:
        raise ValueError("objective_dim must be ≥ 2")

    # ------------ 2-D: closed-form linear spacing -------------------------
    if objective_dim == 2:
        step = 1.0 / (num_base_policies - 1)
        weights = np.array(
            [[1 - i * step, i * step] for i in range(num_base_policies)],
            dtype=np.float32,
        )
        return weights

    # ------------ D >= 3: simplex-lattice --------------------------------
    # smallest H with enough lattice points
    H = 0
    while math.comb(H + objective_dim - 1, objective_dim - 1) < num_base_policies:
        H += 1

    # generate all integer tuples (k1,…,kD) with sum = H
    def lattice_points(H: int, D: int):
        """yield integer tuples that sum to H (stars-and-bars)."""
        for bars in itertools.combinations(range(H + D - 1), D - 1):
            bars = (-1,) + bars + (H + D - 1,)
            ks = [bars[i + 1] - bars[i] - 1 for i in range(D)]
            yield ks

    points = [np.array(ks, dtype=np.float32) for ks in lattice_points(H, objective_dim)]

    if len(points) < num_base_policies:          # shouldn’t happen, but guard
        raise RuntimeError("logic error in lattice generation")

    # deterministic subsample to exactly N points
    if len(points) > num_base_policies:
        if seed is None:
            points = points[:num_base_policies]
        else:  # optional random subsample (reproducible with seed)
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(points), num_base_policies, replace=False)
            points = [points[i] for i in sorted(idx)]

    weights = np.stack(points, axis=0) / float(H)   # normalise to sum=1
    return weights.astype(np.float32)


def two_direction_shift(batch: np.ndarray, shift: float) -> np.ndarray:
    """
    For every base vector in *batch* (shape N×D) return **two** shifted
    vectors -> shape (N, 2, D).

    Example (D=3, shift=0.1):
        [0.4, 0.4, 0.2] ->
            [[0.3, 0.5, 0.2],   # clockwise (+)
             [0.3, 0.4, 0.3]]   # counter-clockwise (−)

    If the base component holds **less** than *shift*, we move that amount in
    the *opposite* sign (add instead of subtract) so no component goes negative.

    """
    N, D = batch.shape
    out = np.empty((N, 2, D), dtype=batch.dtype)

    for n in range(N):
        # print(f"Processing vector {n+1}/{N}: {batch[n]}")  # Debugging output
        w = batch[n].copy()
        i = int(np.argmax(w))             # donor
        j_plus  = (i + 1) % D             # clockwise receiver
        j_minus = (i - 1) % D             # counter-clockwise

        delta = min(shift, w[i])

        # + direction
        w_plus = w.copy()
        if w[i] >= shift:                 # normal case (enough mass)
            w_plus[i]      -= delta
            w_plus[j_plus] += delta
        else:                             # donor too small → add instead
            w_plus[i]      += delta
            w_plus[j_plus] -= delta
        w_plus = np.clip(w_plus, 0.0, None)
        w_plus /= w_plus.sum()


        # − direction
        w_minus = w.copy()
        if w[i] >= shift:
            w_minus[i]       -= delta
            w_minus[j_minus] += delta
        else:
            w_minus[i]       += delta
            w_minus[j_minus] -= delta
        w_minus = np.clip(w_minus, 0.0, None)
        w_minus /= w_minus.sum()

        out[n, 0] = w_plus
        out[n, 1] = w_minus


    return np.unique(out, axis=1) if batch.shape[1] == 2 else out   # shape (N, 1, 2) for 2-D and (N, 2, D) for higher D


def vec_tag(vec, precision: int = 3, max_len: int = 40) -> str:
    """
    Convert a weight vector into a filename-friendly tag.

    Examples (precision=3)
    ----------------------
    [1, 0]                     → w1_0
    [0.9, 0.1]                 → w0p9_0p1
    [0.333333]*3               → w0p333_0p333_0p333
    10-D Dirichlet sample      → w0p22_0p12_0p02_...   (until >max_len -> hash)

    Parameters
    ----------
    vec        : Sequence[float]
    precision  : int   digits after decimal
    max_len    : int   max characters before falling back to hash tag
    """
    parts = []
    for x in vec:
        s = f"{x:.{precision}f}".rstrip("0").rstrip(".")   # 0.600 → 0.6
        s = s.replace(".", "p")                            # 0.6 → 0p6
        parts.append(s)
    tag = "w" + "_".join(parts)

    # Optional shorten
    if len(tag) > max_len:
        h = hashlib.sha1("_".join(parts).encode()).hexdigest()[:8]
        tag = f"w{h}"

    return tag



def get_layer_params(param_dict, weight_key, bias_key):
    """
    param_dict: e.g. model_original_weights
    weight_key: e.g. "mlp_extractor.policy_net.0.weight"
    bias_key:   e.g. "mlp_extractor.policy_net.0.bias"
    
    Returns:
      W, b as NumPy arrays, with W shaped (in_dim, out_dim).
    """
    W_pytorch = param_dict[weight_key]   # shape (out_dim, in_dim) in PyTorch
    b_pytorch = param_dict[bias_key]     # shape (out_dim,)
    
    # Convert to CPU NumPy
    W_np = W_pytorch.cpu().detach().numpy()
    b_np = b_pytorch.cpu().detach().numpy()
    
    # Transpose to (in_dim, out_dim)
    W_np = W_np.T
    # b stays the same shape (out_dim,)
    
    return W_np, b_np


def match_layer_hungarian(WA, WB, bA, bB, cost_metric='l2'):
    """
    Finds an optimal 1-to-1 matching between the neurons of Model A's layer and Model B's layer
    using the Hungarian algorithm to minimize total cost.

    WA, WB: weight matrices for a single layer shape (in_dim, num_neurons).
    bA, bB: bias vectors for the same layer shape (num_neurons,).

    cost_metric: 'l2' or 'l1' or any cost function you define.

    Returns:
      row_ind, col_ind: The assignments from linear_sum_assignment
      total_cost: sum of matched cost for the best assignment
    """
    # Typically we assume shape = (in_dim, num_neurons)
    # We'll interpret "num_neurons" = WA.shape[1].
    # However, be sure your shapes line up with your actual code.
    in_dim, num_neurons = WA.shape

    # Build cost matrix of shape (num_neurons, num_neurons)
    # cost[i, j] = distance between A's i-th neuron and B's j-th neuron
    # We combine the difference of weights and biases.

    cost_matrix = np.zeros((num_neurons, num_neurons))

    for i in range(num_neurons):
        for j in range(num_neurons):
            # Weight difference
            w_diff = WA[:, i] - WB[:, j]
            # Bias difference
            b_diff = bA[i] - bB[j]

            if cost_metric == 'l2':
                cost_val = np.sum(w_diff**2) + b_diff**2
            elif cost_metric == 'l1':
                cost_val = np.sum(np.abs(w_diff)) + abs(b_diff)
            else:
                raise ValueError("Unsupported cost metric.")
            
            cost_matrix[i, j] = cost_val

    # Hungarian algorithm to find minimal-cost matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()

    return row_ind, col_ind, total_cost


def model_distance_hungarian(modelA_params, modelB_params, cost_metric='l2'):
    """
    modelA_params, modelB_params: list of (W, b) for each hidden layer (same shapes), ignoring the final output layer or not, depending on your needs.

    cost_metric: 'l2' or 'l1'
    
    Returns:
      total_matched_cost: sum of best matching costs across all layers
      matches_per_layer: list of (row_ind, col_ind, layer_cost)
    """
    total_matched_cost = 0.0
    matches_per_layer = []

    for (WA, bA), (WB, bB) in zip(modelA_params, modelB_params):
        # WA, WB shape = (in_dim, out_dim)
        row_ind, col_ind, layer_cost = match_layer_hungarian(WA, WB, bA, bB, cost_metric)
        total_matched_cost += layer_cost
        matches_per_layer.append((row_ind, col_ind, layer_cost))

    return total_matched_cost, matches_per_layer




def choose_nondominated_points(points,n_obj, weights=None):
    """
    Choose the nondominated points from a set of points.
    
    Parameters:
    points (ndarray): An array of shape (n_points, n_objectives),
                      where each row represents a point in objective space.
    
    Returns:
    ndarray: An array of nondominated points.
    """
    points = points.reshape(-1, n_obj)
    if weights is not None:
        weights = np.array(weights).reshape(-1, n_obj)
        if points.shape[0] != weights.shape[0]:
            raise ValueError("The number of points and weights must be the same.")
    n_points = points.shape[0]
    is_nondominated = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if is_nondominated[i]:
            pi = points[i]
            is_dominated = np.all(points <= pi, axis=1) & np.any(points < pi, axis=1)
            is_nondominated[is_dominated] = False

    return points[is_nondominated], weights[is_nondominated] if weights is not None else None