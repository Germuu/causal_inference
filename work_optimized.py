import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Dict, Tuple
from math import fsum
from joblib import Parallel, delayed

# -----------------------------
# Count generation (vectorized)
# -----------------------------
def generate_counts_vectorized(
    n_samples: int,
    theta_X: float,
    theta_Y_given_X: Dict[int, float],
    n_trials: int
) -> NDArray[np.int64]:
    theta_Y0, theta_Y1 = theta_Y_given_X[0], theta_Y_given_X[1]
    p = np.array([
        theta_X * theta_Y1,       # Y1,X1
        theta_X * (1 - theta_Y1), # Y0,X1
        (1 - theta_X) * theta_Y0, # Y1,X0
        (1 - theta_X) * (1 - theta_Y0)  # Y0,X0
    ])
    counts = np.random.multinomial(n_samples, p, size=n_trials)
    return counts  # shape (n_trials, 4)

# -----------------------------
# Discretize grid
# -----------------------------
def discretize_params(k: int) -> NDArray[np.float64]:
    step = 2 ** (-k)
    grid = np.arange(step, 1, step, dtype=np.float64)
    return grid

# -----------------------------
# Neighbors (precomputed)
# -----------------------------
def neighbors(val, grid: NDArray[np.float64]) -> list[float]:
    idx = np.searchsorted(grid, val)
    candidates = []
    if idx > 0:
        candidates.append(grid[idx - 1])
    if idx < len(grid):
        candidates.append(grid[min(idx, len(grid)-1)])
    return sorted(set(candidates))

# -----------------------------
# Log-likelihood (unrolled)
# -----------------------------
def log_likelihood_fast(counts: Tuple[int, int, int, int], theta_X, theta_Y0, theta_Y1) -> float:
    n_Y1_X1, n_Y0_X1, n_Y1_X0, n_Y0_X0 = counts
    ll = 0.0
    if n_Y1_X1 > 0: ll += n_Y1_X1 * np.log(theta_Y1)
    if n_Y0_X1 > 0: ll += n_Y0_X1 * np.log(1 - theta_Y1)
    if n_Y1_X0 > 0: ll += n_Y1_X0 * np.log(theta_Y0)
    if n_Y0_X0 > 0: ll += n_Y0_X0 * np.log(1 - theta_Y0)
    n_X1, n_X0 = n_Y1_X1 + n_Y0_X1, n_Y1_X0 + n_Y0_X0
    if n_X1 > 0: ll += n_X1 * np.log(theta_X)
    if n_X0 > 0: ll += n_X0 * np.log(1 - theta_X)
    return ll

# -----------------------------
# Fit direction
# -----------------------------
def fit_direction_from_counts(
    counts: Tuple[int, int, int, int],
    grid: NDArray[np.float64],
    direction: str = "X->Y"
) -> Tuple[Tuple[float,float,float], float]:

    if direction == "Y->X":
        n_Y1, n_Y0 = counts[0] + counts[2], counts[1] + counts[3]
        n_X1_Y0, n_X0_Y0 = counts[1], counts[3]
        n_X1_Y1, n_X0_Y1 = counts[0], counts[2]
        counts = (n_X1_Y1, n_X0_Y1, n_X1_Y0, n_X0_Y0)
    n_Y1_X1, n_Y0_X1, n_Y1_X0, n_Y0_X0 = counts

    # MLE
    n_X1, n_X0 = n_Y1_X1 + n_Y0_X1, n_Y1_X0 + n_Y0_X0
    mle_theta_X = n_X1 / (n_X1 + n_X0) if n_X1 + n_X0 > 0 else 0.0
    mle_theta_Y0 = n_Y1_X0 / (n_Y1_X0 + n_Y0_X0) if n_Y1_X0 + n_Y0_X0 > 0 else 0.0
    mle_theta_Y1 = n_Y1_X1 / (n_Y1_X1 + n_Y0_X1) if n_Y1_X1 + n_Y0_X1 > 0 else 0.0

    theta_X_candidates = neighbors(mle_theta_X, grid)
    theta_Y0_candidates = neighbors(mle_theta_Y0, grid)
    theta_Y1_candidates = neighbors(mle_theta_Y1, grid)

    best_ll = -np.inf
    best_params = (0.0,0.0,0.0)
    for theta_X, theta_Y0, theta_Y1 in product(theta_X_candidates, theta_Y0_candidates, theta_Y1_candidates):
        ll = log_likelihood_fast(counts, theta_X, theta_Y0, theta_Y1)
        if ll > best_ll:
            best_ll = ll
            best_params = (theta_X, theta_Y0, theta_Y1)
    return best_params, best_ll

# -----------------------------
# Single trial
# -----------------------------
def single_trial(n_samples, grid, k):
    theta_X = np.random.choice(grid)
    theta_Y0 = np.random.choice(grid)
    theta_Y1 = np.random.choice(grid)
    counts = generate_counts_vectorized(n_samples, theta_X, {0:theta_Y0,1:theta_Y1}, 1)[0]
    ll_XY = fit_direction_from_counts(counts, grid, "X->Y")[1]
    ll_YX = fit_direction_from_counts(counts, grid, "Y->X")[1]
    if ll_XY > ll_YX:
        return 1,0
    elif ll_XY == ll_YX:
        return 0.5,1
    else:
        return 0,0

# -----------------------------
# Run experiment
# -----------------------------
def run_experiment(n_trials=100000, sample_sizes=None, k_values=None):
    if sample_sizes is None:
        sample_sizes = [50,100,250,500,1000,2500,5000,10000,20000,40000]
    if k_values is None:
        k_values = range(2,10)

    results = {}
    grids = {k: discretize_params(k) for k in k_values}

    for k in k_values:
        grid = grids[k]
        print(f"Running for k={k}")
        for n_samples in sample_sizes:
            trial_results = Parallel(n_jobs=-1)(
                delayed(single_trial)(n_samples, grid, k) for _ in range(n_trials)
            )
            wins, ties = map(sum, zip(*trial_results))
            prop_correct = wins / n_trials
            prop_ties = ties / n_trials
            results[(k,n_samples)] = {"accuracy": prop_correct, "ties": prop_ties}
            print(f"  n={n_samples}: Acc={prop_correct:.3f}, Ties={prop_ties:.3f}")
    return results, sample_sizes, k_values

# -----------------------------
# Plot results
# -----------------------------
def plot_results(results, sample_sizes, k_values):
    plt.figure(figsize=(12,7))
    for k in k_values:
        accuracies = [results[(k,n)]["accuracy"] for n in sample_sizes]
        plt.plot(sample_sizes, accuracies, marker="o", label=f"k={k}")
    plt.xscale("log")
    plt.xlabel("Sample size (log scale)")
    plt.ylabel("Proportion correct (X→Y wins)")
    plt.title("Causal direction identification accuracy vs Sample size")
    plt.legend(title="Discretization k")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,7))
    for k in k_values:
        tie_rates = [results[(k,n)]["ties"] for n in sample_sizes]
        plt.plot(sample_sizes, tie_rates, marker="s", linestyle="--", label=f"k={k}")
    plt.xscale("log")
    plt.xlabel("Sample size (log scale)")
    plt.ylabel("Proportion of ties")
    plt.title("Proportion of likelihood ties vs Sample size")
    plt.legend(title="Discretization k")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main
# -----------------------------
if __name__=="__main__":
    np.random.seed(42)
    results, sample_sizes, k_values = run_experiment(n_trials=5000)  # adjust n_trials for speed
    plot_results(results, sample_sizes, k_values)
