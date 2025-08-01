import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def generate_counts_fast(
    n_samples: int, theta_X: float, theta_Y_given_X: dict
) -> tuple[int, int, int, int, int, int]:
    theta_Y0 = theta_Y_given_X[0]
    theta_Y1 = theta_Y_given_X[1]

    p_00 = (1 - theta_X) * (1 - theta_Y0)
    p_01 = (1 - theta_X) * theta_Y0
    p_10 = theta_X * (1 - theta_Y1)
    p_11 = theta_X * theta_Y1

    counts = np.random.multinomial(n_samples, [p_11, p_10, p_01, p_00])

    n_Y1_X1 = counts[0]
    n_Y0_X1 = counts[1]
    n_Y1_X0 = counts[2]
    n_Y0_X0 = counts[3]
    n_X1 = n_Y1_X1 + n_Y0_X1
    n_X0 = n_Y1_X0 + n_Y0_X0

    return n_X1, n_X0, n_Y1_X0, n_Y0_X0, n_Y1_X1, n_Y0_X1


def discretize_params(k: int) -> NDArray[np.float64]:
    step: float = 2 ** (-k)
    return np.arange(start=0, stop=1 + step, step=step, dtype=np.float64)


def log_likelihood_fast(
    counts: tuple[int, int, int, int, int, int],
    theta_X: float,
    theta_Y0: float,
    theta_Y1: float,
) -> float:
    eps = 1e-12

    n_X1, n_X0, n_Y1_X0, n_Y0_X0, n_Y1_X1, n_Y0_X1 = counts

    ll_X = n_X1 * np.log(theta_X + eps) + n_X0 * np.log(1 - theta_X + eps)
    ll_Y0 = n_Y1_X0 * np.log(theta_Y0 + eps) + n_Y0_X0 * np.log(1 - theta_Y0 + eps)
    ll_Y1 = n_Y1_X1 * np.log(theta_Y1 + eps) + n_Y0_X1 * np.log(1 - theta_Y1 + eps)

    return ll_X + ll_Y0 + ll_Y1


def neighbors(val, grid: NDArray[np.float64]) -> list[float]:
    idx: int = np.searchsorted(grid, val)
    candidates: list[float] = []
    if idx > 0:
        candidates.append(grid[idx - 1])
    if idx < len(grid):
        candidates.append(grid[min(idx, len(grid) - 1)])
    return list(set(candidates))


def fit_direction_from_counts(
    counts: tuple[int, int, int, int, int, int],
    grid: NDArray[np.float64],
    direction: str = "X->Y",
) -> tuple[tuple[float, float, float], float]:
    if direction == "X->Y":
        n_X1, n_X0, n_Y1_X0, n_Y0_X0, n_Y1_X1, n_Y0_X1 = counts
    else:  # Y->X: just swap labels
        n_Y1 = counts[4] + counts[2]
        n_Y0 = counts[5] + counts[3]
        n_X1_Y0 = counts[5]
        n_X0_Y0 = counts[3]
        n_X1_Y1 = counts[4]
        n_X0_Y1 = counts[2]
        counts = (n_Y1, n_Y0, n_X1_Y0, n_X0_Y0, n_X1_Y1, n_X0_Y1)

        n_X1, n_X0, n_Y1_X0, n_Y0_X0, n_Y1_X1, n_Y0_X1 = counts

    mle_theta_X = n_X1 / (n_X1 + n_X0) if (n_X1 + n_X0) > 0 else 0.0
    mle_theta_Y0 = n_Y1_X0 / (n_Y1_X0 + n_Y0_X0) if (n_Y1_X0 + n_Y0_X0) > 0 else 0.0
    mle_theta_Y1 = n_Y1_X1 / (n_Y1_X1 + n_Y0_X1) if (n_Y1_X1 + n_Y0_X1) > 0 else 0.0

    theta_X_candidates = neighbors(mle_theta_X, grid)
    theta_Y0_candidates = neighbors(mle_theta_Y0, grid)
    theta_Y1_candidates = neighbors(mle_theta_Y1, grid)

    best_ll = -np.inf
    best_params = (0.0, 0.0, 0.0)

    for theta_X, theta_Y0, theta_Y1 in product(
        theta_X_candidates, theta_Y0_candidates, theta_Y1_candidates
    ):
        ll = log_likelihood_fast(
            counts, theta_X=theta_X, theta_Y0=theta_Y0, theta_Y1=theta_Y1
        )
        if ll > best_ll:
            best_ll = ll
            best_params = (theta_X, theta_Y0, theta_Y1)

    return best_params, best_ll


def run_experiment(
    n_trials=1000, sample_sizes=None, k_values=None
) -> tuple[dict[tuple[int, int], float], list[int], range]:
    if sample_sizes is None:
        sample_sizes = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000, 40000]
    if k_values is None:
        k_values = range(2, 10)

    results: dict[tuple[int, int], float] = {}

    grids: dict[int, NDArray[np.float64]] = {k: discretize_params(k) for k in k_values}

    for k in k_values:
        grid = grids[k]
        print(f"Running for k={k}")
        for n_samples in sample_sizes:
            wins = 0
            for _ in range(n_trials):
                theta_X = np.random.choice(grid)
                theta_Y0 = np.random.choice(grid)
                theta_Y1 = np.random.choice(grid)
                theta_Y_given_X = {0: theta_Y0, 1: theta_Y1}

                counts = generate_counts_fast(n_samples, theta_X, theta_Y_given_X)

                ll_XY = fit_direction_from_counts(counts, grid, "X->Y")[1]
                ll_YX = fit_direction_from_counts(counts, grid, "Y->X")[1]

                if ll_XY > ll_YX:
                    wins += 1

            proportion = wins / n_trials
            results[(k, n_samples)] = proportion
            print(f"  n={n_samples}: Prop correct (X->Y) = {proportion:.3f}")

    return results, sample_sizes, k_values


def plot_results(results, sample_sizes, k_values) -> None:
    plt.figure(figsize=(12, 7))

    for k in k_values:
        proportions = [results[(k, n)] for n in sample_sizes]
        plt.plot(sample_sizes, proportions, marker="o", label=f"k={k}")

    plt.xscale("log")
    plt.xlabel("Sample size (log scale)")
    plt.ylabel("Proportion correct (X→Y wins)")
    plt.title("Causal direction identification accuracy vs Sample size")
    plt.legend(title="Discretization granularity k")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    results, sample_sizes, k_values = run_experiment()
    plot_results(results, sample_sizes, k_values)
