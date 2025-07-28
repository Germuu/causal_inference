import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from typing import Callable


def generate_data_continuous(
    n_samples: int,
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    noise_sampler: Callable[[int], np.ndarray],
    x_sampler: Callable[[int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    X = x_sampler(n_samples)
    N = noise_sampler(n_samples)
    Y = f(X, N)
    return X, Y


def log_likelihood_gaussian(Y_true: np.ndarray, Y_pred: np.ndarray, noise_std: float = 1.0) -> float:
    residual = Y_true - Y_pred
    n = len(Y_true)
    ll = -0.5 * n * np.log(2 * np.pi * noise_std**2) - (np.sum(residual**2) / (2 * noise_std**2))
    return ll


def fit_direction_continuous(X: np.ndarray, Y: np.ndarray) -> float:
    model = LinearRegression().fit(X.reshape(-1, 1), Y)
    Y_pred = model.predict(X.reshape(-1, 1))
    noise_std = np.std(Y - Y_pred)
    return log_likelihood_gaussian(Y_true=Y, Y_pred=Y_pred, noise_std=noise_std)


def run_experiment_continuous(
    n_trials: int = 100,
    sample_sizes: list[int] = None,
    f: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
    x_sampler: Callable[[int], np.ndarray] = None,
    noise_sampler: Callable[[int], np.ndarray] = None
) -> dict[int, float]:
    if sample_sizes is None:
        sample_sizes = [50, 100, 500, 1000, 5000]

    if f is None:
        f = lambda x, n: np.sin(x) + n
    if x_sampler is None:
        x_sampler = lambda n: np.random.uniform(-2, 2, n)
    if noise_sampler is None:
        noise_sampler = lambda n: np.random.normal(0, 0.5, n)

    results = {}

    for n_samples in sample_sizes:
        wins = 0
        for _ in range(n_trials):
            X, Y = generate_data_continuous(n_samples, f, noise_sampler, x_sampler)

            ll_XY = fit_direction_continuous(X, Y)
            ll_YX = fit_direction_continuous(Y, X)

            if ll_XY > ll_YX:
                wins += 1

        proportion = wins / n_trials
        results[n_samples] = proportion
        print(f"Sample size {n_samples}: Proportion correct (X->Y) = {proportion:.3f}")

    return results


def plot_results_continuous(results: dict[int, float]) -> None:
    sample_sizes = list(results.keys())
    proportions = list(results.values())

    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, proportions, marker="o")
    plt.xscale("log")
    plt.xlabel("Sample size (log scale)")
    plt.ylabel("Proportion correct (X → Y wins)")
    plt.title("Causal Direction Detection Accuracy (Functional Model)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    # You can customize the functional relationship here
    f = lambda x, n: np.sin(x) + n
    x_sampler = lambda n: np.random.uniform(-2, 2, n)
    noise_sampler = lambda n: np.random.normal(0, 0.5, n)

    results = run_experiment_continuous(
        n_trials=100,
        sample_sizes=[50, 100, 500, 1000, 5000],
        f=f,
        x_sampler=x_sampler,
        noise_sampler=noise_sampler
    )

    plot_results_continuous(results)
