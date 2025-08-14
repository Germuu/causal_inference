import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Dict, Tuple


def generate_counts_multinomial(
    n_samples: int,
    theta_X: NDArray[np.float64],
    theta_Y_given_X: NDArray[np.float64],
) -> NDArray[np.int64]:
    """
    Generate counts for X and Y, both taking 4 possible values.
    theta_X: shape (4,) marginal probabilities for X
    theta_Y_given_X: shape (4,4) conditional probabilities P(Y|X)
    Returns counts: shape (4,4), counts[x, y]
    """
    # Joint probabilities
    p_XY = theta_X[:, None] * theta_Y_given_X  # shape (4,4)
    p_XY = p_XY.flatten()
    counts = np.random.multinomial(n_samples, p_XY)
    return counts.reshape(4, 4)


def log_likelihood_multinomial(
    counts: NDArray[np.int64],
    theta_X: NDArray[np.float64],
    theta_Y_given_X: NDArray[np.float64]
) -> float:
    joint_prob = theta_X[:, None] * theta_Y_given_X
    with np.errstate(divide='ignore'):
        ll_matrix = counts * np.log(joint_prob)
    ll_matrix = np.where(counts > 0, ll_matrix, 0.0)
    return ll_matrix.sum()


def discretize_params(k: int) -> NDArray[np.float64]:
    step = 2 ** (-k)
    grid = np.arange(start=0, stop=1 + step, step=step, dtype=np.float64)
    return grid[(grid > 0.0) & (grid < 1.0)]


def fit_direction_from_counts_multinomial(
    counts: NDArray[np.int64],
    grid: NDArray[np.float64],
    direction: str = "X->Y"
) -> Tuple[Tuple[NDArray[np.float64], NDArray[np.float64]], float]:
    """
    Fit MLE parameters for 4x4 counts and return log-likelihood.
    Uses discretization grid to snap MLEs to nearby values.
    """
    if direction == "Y->X":
        counts = counts.T  # transpose for reverse direction

    # MLE for X
    theta_X_mle = counts.sum(axis=1) / counts.sum()

    # MLE for Y|X
    theta_Y_given_X_mle = np.zeros_like(counts, dtype=np.float64)
    for x in range(4):
        total = counts[x].sum()
        if total > 0:
            theta_Y_given_X_mle[x] = counts[x] / total
        else:
            theta_Y_given_X_mle[x] = 0.25  # uniform fallback

    # Snap to nearest grid points
    theta_X_candidates = [theta_X_mle, *[g for g in grid if np.abs(g - theta_X_mle).min() < 1e-8]]
    theta_Y_candidates = np.vectorize(lambda v: grid[np.argmin(np.abs(grid - v))])(theta_Y_given_X_mle)

    # Log-likelihood at snapped MLE
    ll = log_likelihood_multinomial(counts, theta_X_candidates[0], theta_Y_candidates)
    return (theta_X_candidates[0], theta_Y_candidates), ll


def run_experiment_multinomial(
    n_trials=1000, sample_sizes=None, k_values=None
) -> Tuple[Dict[Tuple[int, int], Dict[str, float]], list[int], range]:
    if sample_sizes is None:
        sample_sizes = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000, 40000]
    if k_values is None:
        k_values = range(2, 6)

    results: dict = {}
    grids = {k: discretize_params(k) for k in k_values}

    for k in k_values:
        grid = grids[k]
        print(f"Running for k={k}")
        for n_samples in sample_sizes:
            wins = 0
            ties = 0
            for _ in range(n_trials):
                theta_X = np.random.choice(grid, size=4)
                theta_X /= theta_X.sum()  # normalize

                theta_Y_given_X = np.zeros((4, 4))
                for x in range(4):
                    y_probs = np.random.choice(grid, size=4)
                    y_probs /= y_probs.sum()
                    theta_Y_given_X[x] = y_probs

                counts = generate_counts_multinomial(n_samples, theta_X, theta_Y_given_X)

                ll_XY = fit_direction_from_counts_multinomial(counts, grid, "X->Y")[1]
                ll_YX = fit_direction_from_counts_multinomial(counts, grid, "Y->X")[1]

                if ll_XY > ll_YX:
                    wins += 1
                elif ll_XY == ll_YX:
                    wins += 0.5
                    ties += 1

            results[(k, n_samples)] = {
                "accuracy": wins / n_trials,
                "ties": ties / n_trials
            }
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
    plt.title("Causal direction identification accuracy vs Sample size (4-valued multinomial)")
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
    plt.title("Proportion of likelihood ties vs Sample size (4-valued multinomial)")
    plt.legend(title="Discretization granularity k")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    results, sample_sizes, k_values = run_experiment_multinomial()
    plot_results(results, sample_sizes, k_values)
