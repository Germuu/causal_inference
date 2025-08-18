import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Dict, Tuple
from scipy.spatial import cKDTree
from joblib import Parallel, delayed

# ------------------ Core Functions ------------------

def generate_counts_fast_4val(
    n_samples: int,
    theta_X: NDArray[np.float64],
    theta_Y_given_X: Dict[int, NDArray[np.float64]],
) -> NDArray[np.int64]:
    joint_probs = (theta_X[:, None] * np.vstack([theta_Y_given_X[x] for x in range(4)])).ravel()
    counts = np.random.multinomial(n_samples, joint_probs).reshape(4, 4)
    return counts


def discretize_params_simplex(k: int, dim: int = 4) -> NDArray[np.float64]:
    step = 2 ** (-k)
    valid_points = []

    def generate_simplex_points(remaining_sum, remaining_dims, current_point):
        if remaining_dims == 1:
            if 0 < remaining_sum < 1:
                valid_points.append(current_point + [remaining_sum])
            return
        max_val = min(1 - step, remaining_sum - (remaining_dims - 1) * step)
        for val in np.arange(step, max_val + step, step):
            generate_simplex_points(
                remaining_sum - val,
                remaining_dims - 1,
                current_point + [val]
            )

    generate_simplex_points(1.0, dim, [])
    return np.array(valid_points)


def log_likelihood_fast_4val(
    counts: NDArray[np.int64],
    theta_X: NDArray[np.float64],
    theta_Y_given_X: Dict[int, NDArray[np.float64]],
) -> float:
    marginal_X = np.sum(counts, axis=1)
    ll = np.sum(marginal_X[marginal_X > 0] * np.log(theta_X[marginal_X > 0]))

    theta_Y = np.vstack([theta_Y_given_X[x] for x in range(4)])
    mask = counts > 0
    ll += np.sum(counts[mask] * np.log(theta_Y[mask]))
    return ll


def mle_estimate_4val(counts: NDArray[np.int64]) -> Tuple[NDArray[np.float64], Dict[int, NDArray[np.float64]]]:
    n_total = np.sum(counts)
    marginal_X = np.sum(counts, axis=1)
    theta_X_mle = marginal_X / n_total
    theta_Y_given_X_mle = {}
    for x in range(4):
        theta_Y_given_X_mle[x] = counts[x, :] / marginal_X[x] if marginal_X[x] > 0 else np.ones(4) / 4
    return theta_X_mle, theta_Y_given_X_mle


def neighbors_simplex(val: NDArray[np.float64], tree: cKDTree, tolerance: float = 1e-6) -> list[NDArray[np.float64]]:
    idx = tree.query_ball_point(val, tolerance)
    if len(idx) == 0:
        idx = [tree.query(val)[1]]
    return tree.data[idx]


def fit_direction_from_counts_4val_worker(counts, grid, direction="X->Y", k=4):
    tree = cKDTree(grid)
    if direction == "Y->X":
        counts = counts.T

    theta_X_mle, theta_Y_given_X_mle = mle_estimate_4val(counts)
    theta_X_candidates = neighbors_simplex(theta_X_mle, tree)
    theta_Y_given_X = {x: neighbors_simplex(theta_Y_given_X_mle[x], tree)[0] for x in range(4)}

    best_ll = -np.inf
    best_params = None
    for theta_X in theta_X_candidates:
        ll = log_likelihood_fast_4val(counts, theta_X, theta_Y_given_X)
        if ll > best_ll:
            best_ll = ll
            best_params = (theta_X, theta_Y_given_X)

    return best_params, best_ll


def run_experiment_4val(n_trials=10000, sample_sizes=None, k_values=None, n_jobs=1):
    if sample_sizes is None:
        sample_sizes = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000, 40000]
    if k_values is None:
        k_values = range(2, 6)

    results = {}
    grids = {k: discretize_params_simplex(k, 4) for k in k_values}

    for k in k_values:
        grid = grids[k]
        print(f"Running for k={k}")

        def single_trial(n_samples):
            theta_X = grid[np.random.randint(len(grid))]
            theta_Y_given_X = {x: grid[np.random.randint(len(grid))] for x in range(4)}
            counts = generate_counts_fast_4val(n_samples, theta_X, theta_Y_given_X)
            try:
                ll_XY = fit_direction_from_counts_4val_worker(counts, grid, "X->Y", k)[1]
                ll_YX = fit_direction_from_counts_4val_worker(counts, grid, "Y->X", k)[1]
                if ll_XY > ll_YX:
                    return 1, 0
                elif ll_XY == ll_YX:
                    return 0.5, 1
                else:
                    return 0, 0
            except:
                return 0, 0

        for n_samples in sample_sizes:
            if n_jobs == 1:
                trial_results = [single_trial(n_samples) for _ in range(n_trials)]
            else:
                trial_results = Parallel(n_jobs=n_jobs)(
                    delayed(single_trial)(n_samples) for _ in range(n_trials)
                )

            wins = sum(r[0] for r in trial_results)
            ties = sum(r[1] for r in trial_results)
            results[(k, n_samples)] = {"accuracy": wins / n_trials, "ties": ties / n_trials}
            print(f"  n={n_samples}: Acc={wins/n_trials:.3f}, Ties={ties/n_trials:.3f}")

    return results, sample_sizes, k_values


def plot_results(results, sample_sizes, k_values):
    plt.figure(figsize=(12, 7))
    for k in k_values:
        accuracies = [results[(k, n)]["accuracy"] for n in sample_sizes]
        plt.plot(sample_sizes, accuracies, marker="o", label=f"k={k}")
    plt.xscale("log")
    plt.xlabel("Sample size (log scale)")
    plt.ylabel("Proportion correct (X→Y wins)")
    plt.title("Causal direction identification accuracy vs Sample size")
    plt.legend(title="Discretization granularity k")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 7))
    for k in k_values:
        tie_rates = [results[(k, n)]["ties"] for n in sample_sizes]
        plt.plot(sample_sizes, tie_rates, marker="s", linestyle="--", label=f"k={k}")
    plt.xscale("log")
    plt.xlabel("Sample size (log scale)")
    plt.ylabel("Proportion of ties")
    plt.title("Proportion of likelihood ties vs Sample size")
    plt.legend(title="Discretization granularity k")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ------------------ Main Execution ------------------

if __name__ == "__main__":
    np.random.seed(42)
    results, sample_sizes, k_values = run_experiment_4val(n_trials=1000, n_jobs=-1)  # parallel
    plot_results(results, sample_sizes, k_values)
