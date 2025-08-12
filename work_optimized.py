import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Dict, Tuple


def generate_counts_fast(
    n_samples: int,
    theta_X: float,
    theta_Y_given_X: Dict[int, float],
) -> Tuple[int, int, int, int, int, int]:
    theta_Y0: float = theta_Y_given_X[0]
    theta_Y1: float = theta_Y_given_X[1]

    p_00: float = (1 - theta_X) * (1 - theta_Y0)
    p_01: float = (1 - theta_X) * theta_Y0
    p_10: float = theta_X * (1 - theta_Y1)
    p_11: float = theta_X * theta_Y1

    counts: np.ndarray = np.random.multinomial(n_samples, [p_11, p_10, p_01, p_00])

    n_Y1_X1: int = counts[0]
    n_Y0_X1: int = counts[1]
    n_Y1_X0: int = counts[2]
    n_Y0_X0: int = counts[3]
    n_X1: int = n_Y1_X1 + n_Y0_X1
    n_X0: int = n_Y1_X0 + n_Y0_X0

    return n_X1, n_X0, n_Y1_X0, n_Y0_X0, n_Y1_X1, n_Y0_X1


def discretize_params(k: int) -> NDArray[np.float64]:
    step: float = 2 ** (-k)
    grid = np.arange(start=0, stop=1 + step, step=step, dtype=np.float64)
    # Exclude boundary points 0 and 1
    return grid[(grid > 0.0) & (grid < 1.0)]


def log_likelihood_fast(
    counts: tuple[int, int, int, int, int, int],
    theta_X: float,
    theta_Y0: float,
    theta_Y1: float,
) -> float:
    params: list[float] = [
        theta_X,        # n_X1
        1 - theta_X,    # n_X0
        theta_Y0,       # n_Y1_X0
        1 - theta_Y0,   # n_Y0_X0
        theta_Y1,       # n_Y1_X1
        1 - theta_Y1,   # n_Y0_X1
    ]
    return sum(
        c * np.log(p)
        for c, p in zip(counts, params)
        if c > 0 and p > 0
    )


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
        n_Y1: int = counts[4] + counts[2]
        n_Y0: int = counts[5] + counts[3]
        n_X1_Y0: int = counts[5]
        n_X0_Y0: int = counts[3]
        n_X1_Y1: int = counts[4]
        n_X0_Y1: int = counts[2]
        counts = (n_Y1, n_Y0, n_X1_Y0, n_X0_Y0, n_X1_Y1, n_X0_Y1)

        n_X1, n_X0, n_Y1_X0, n_Y0_X0, n_Y1_X1, n_Y0_X1 = counts

    mle_theta_X: float = n_X1 / (n_X1 + n_X0) if (n_X1 + n_X0) > 0 else 0.0
    mle_theta_Y0: float = n_Y1_X0 / (n_Y1_X0 + n_Y0_X0) if (n_Y1_X0 + n_Y0_X0) > 0 else 0.0
    mle_theta_Y1: float = n_Y1_X1 / (n_Y1_X1 + n_Y0_X1) if (n_Y1_X1 + n_Y0_X1) > 0 else 0.0

    theta_X_candidates: list[float] = neighbors(mle_theta_X, grid)
    theta_Y0_candidates: list[float] = neighbors(mle_theta_Y0, grid)
    theta_Y1_candidates: list[float] = neighbors(mle_theta_Y1, grid)

    best_ll = -np.inf
    best_params = (0.0, 0.0, 0.0)

    for theta_X, theta_Y0, theta_Y1 in product(
        theta_X_candidates, theta_Y0_candidates, theta_Y1_candidates
    ):
        ll: float = log_likelihood_fast(
            counts=counts, theta_X=theta_X, theta_Y0=theta_Y0, theta_Y1=theta_Y1
        )
        if ll > best_ll:
            best_ll: float = ll
            best_params: Tuple[float, float, float] = (theta_X, theta_Y0, theta_Y1)

    return best_params, best_ll


def run_experiment(
    n_trials=10000, sample_sizes=None, k_values=None
) -> tuple[dict[tuple[int, int], dict[str, float]], list[int], range]:
    if sample_sizes is None:
        sample_sizes = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000, 40000]
    if k_values is None:
        k_values = range(2, 10)

    results: dict[tuple[int, int], dict[str, float]] = {}

    grids: dict[int, NDArray[np.float64]] = {k: discretize_params(k) for k in k_values}

    for k in k_values:
        grid = grids[k]

        print(f"Running for k={k}")
        for n_samples in sample_sizes:
            wins = 0
            ties = 0
            for _ in range(n_trials):
                theta_X: float = np.random.choice(grid)
                theta_Y0: float = np.random.choice(grid)
                theta_Y1: float = np.random.choice(grid)
                theta_Y_given_X: Dict[int, float] = {0: theta_Y0, 1: theta_Y1}

                counts: tuple[int, int, int, int, int, int] = generate_counts_fast(n_samples, theta_X, theta_Y_given_X)

                ll_XY: float = fit_direction_from_counts(counts, grid, "X->Y")[1]
                ll_YX: float = fit_direction_from_counts(counts, grid, "Y->X")[1]

                if ll_XY > ll_YX:
                    wins += 1
                elif ll_XY == ll_YX:
                    wins += 0.5
                    ties += 1

            prop_correct: float = wins / n_trials
            prop_ties: float = ties / n_trials
            results[(k, n_samples)] = {"accuracy": prop_correct, "ties": prop_ties}
            print(f"  n={n_samples}: Acc={prop_correct:.3f}, Ties={prop_ties:.3f}")

    return results, sample_sizes, k_values


def plot_results(results, sample_sizes, k_values) -> None:
    # Accuracy plot
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

    # Tie proportion plot
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


if __name__ == "__main__":
    np.random.seed(42)
    results, sample_sizes, k_values = run_experiment()
    plot_results(results, sample_sizes, k_values)
