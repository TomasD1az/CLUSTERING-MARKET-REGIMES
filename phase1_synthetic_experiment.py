"""
Phase-1 synthetic checkpoint for temporally regularized OT regime clustering.

Goal:
- Verify OT geometry separates distributional regimes (same mean/variance).
- Verify temporal penalty reduces fragmentation.
"""

import os
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt

from wasserstein_kmeans import (
    WassersteinKMeans,
    TemporalWassersteinKMeans,
    create_sliding_windows
)
from visualization import plot_historical_coloring


def generate_regime_sequence(
    n_steps: int,
    regimes: list,
    block_min: int,
    block_max: int,
    rng: np.random.Generator
) -> tuple:
    labels = np.zeros(n_steps, dtype=int)
    intervals = []
    t = 0
    prev = None

    while t < n_steps:
        if prev is None or len(regimes) == 1:
            regime = int(rng.choice(regimes))
        else:
            choices = [r for r in regimes if r != prev]
            regime = int(rng.choice(choices))

        block_len = int(rng.integers(block_min, block_max + 1))
        end = min(t + block_len, n_steps)
        labels[t:end] = regime
        intervals.append((t, end))
        prev = regime
        t = end

    return labels, intervals


def sample_regime_returns(regime: int, size: int, rng: np.random.Generator) -> np.ndarray:
    if regime == 0:
        return rng.normal(0.0, 1.0, size)

    if regime == 1:
        df = 4
        scale = np.sqrt((df - 2) / df)
        return rng.standard_t(df, size) * scale

    # Skewed mixture with mean 0 and variance 1
    weight = 0.8
    mean_1 = -0.3
    mean_2 = 1.2
    std = 0.8

    mix = rng.random(size) < weight
    samples = np.empty(size)
    samples[mix] = rng.normal(mean_1, std, np.sum(mix))
    samples[~mix] = rng.normal(mean_2, std, np.sum(~mix))
    return samples


def generate_synthetic_returns(
    n_steps: int = 3000,
    block_min: int = 100,
    block_max: int = 300,
    random_state: int = 42
) -> tuple:
    rng = np.random.default_rng(random_state)

    regimes = [0, 1, 2]
    labels, intervals = generate_regime_sequence(
        n_steps, regimes, block_min, block_max, rng
    )

    returns = np.zeros(n_steps)
    for start, end in intervals:
        regime = labels[start]
        returns[start:end] = sample_regime_returns(regime, end - start, rng)

    return returns, labels, intervals


def standardize_windows(windows: list, eps: float = 1e-8) -> list:
    standardized = []
    for window in windows:
        mean = np.mean(window)
        std = np.std(window)
        if std < eps:
            standardized.append(window - mean)
        else:
            standardized.append((window - mean) / std)
    return standardized


def compute_transition_entropy(labels: np.ndarray, n_states: int) -> float:
    counts = np.zeros((n_states, n_states), dtype=float)
    for t in range(1, len(labels)):
        counts[labels[t - 1], labels[t]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    probs = counts / np.maximum(row_sums, 1.0)

    entropies = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    weights = row_sums[:, 0] / np.maximum(np.sum(row_sums), 1.0)

    return float(np.sum(entropies * weights))


def compute_persistence_metrics(labels: np.ndarray, n_states: int) -> dict:
    switches = int(np.sum(labels[1:] != labels[:-1]))
    switch_freq = switches / max(len(labels) - 1, 1)

    run_lengths = []
    run = 1
    for t in range(1, len(labels)):
        if labels[t] == labels[t - 1]:
            run += 1
        else:
            run_lengths.append(run)
            run = 1
    run_lengths.append(run)

    mean_run = float(np.mean(run_lengths))
    transition_entropy = compute_transition_entropy(labels, n_states)

    return {
        "mean_run_length": mean_run,
        "switch_frequency": float(switch_freq),
        "transition_entropy": transition_entropy
    }


def window_majority_labels(true_labels: np.ndarray, h1: int, h2: int, n_states: int) -> np.ndarray:
    labels = []
    i = 0
    while i + h1 <= len(true_labels):
        window = true_labels[i:i + h1]
        counts = np.bincount(window, minlength=n_states)
        labels.append(int(np.argmax(counts)))
        i += h2
    return np.array(labels, dtype=int)


def best_label_accuracy(pred: np.ndarray, true: np.ndarray, n_states: int) -> float:
    best = 0.0
    for perm in permutations(range(n_states)):
        mapped = np.array([perm[p] for p in pred], dtype=int)
        acc = float(np.mean(mapped == true))
        if acc > best:
            best = acc
    return best


def run_phase1_synthetic(
    n_steps: int = 3000,
    block_min: int = 100,
    block_max: int = 300,
    h1: int = 20,
    stride_values: list = None,
    n_clusters: int = 3,
    lambda_switch: float = 0.5,
    random_state: int = 42,
    output_dir: str = "figures"
) -> None:
    if stride_values is None:
        stride_values = [1, 20]

    os.makedirs(output_dir, exist_ok=True)

    returns, true_labels, regime_intervals = generate_synthetic_returns(
        n_steps=n_steps,
        block_min=block_min,
        block_max=block_max,
        random_state=random_state
    )
    true_state_count = 3

    prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(returns)]))

    for h2 in stride_values:
        windows = create_sliding_windows(returns, h1, h2)
        windows = standardize_windows(windows)

        wk_model = WassersteinKMeans(
            n_clusters=n_clusters,
            p=1,
            max_iter=50,
            tol=1e-6,
            n_init=5,
            random_state=random_state
        )
        wk_model.fit(windows)
        wk_labels = wk_model.labels_

        temporal_model = TemporalWassersteinKMeans(
            n_clusters=n_clusters,
            p=1,
            max_iter=50,
            tol=1e-6,
            n_init=5,
            lambda_switch=lambda_switch,
            random_state=random_state
        )
        temporal_model.fit(windows)
        temporal_labels = temporal_model.labels_

        true_window_labels = window_majority_labels(true_labels, h1, h2, true_state_count)

        wk_metrics = compute_persistence_metrics(wk_labels, n_clusters)
        temporal_metrics = compute_persistence_metrics(temporal_labels, n_clusters)

        wk_acc = None
        temporal_acc = None
        if n_clusters == true_state_count:
            wk_acc = best_label_accuracy(wk_labels, true_window_labels, n_clusters)
            temporal_acc = best_label_accuracy(temporal_labels, true_window_labels, n_clusters)

        print("=" * 70)
        print(f"Synthetic Phase-1 | stride={h2} | K={n_clusters} | lambda={lambda_switch}")
        if wk_acc is not None:
            print("WK-means metrics:", wk_metrics, f"| best acc: {wk_acc:.3f}")
            print("Temporal OT metrics:", temporal_metrics, f"| best acc: {temporal_acc:.3f}")
        else:
            print("WK-means metrics:", wk_metrics, "| best acc: n/a")
            print("Temporal OT metrics:", temporal_metrics, "| best acc: n/a")

        wk_fig = plot_historical_coloring(
            prices,
            wk_labels,
            h1,
            h2,
            regime_intervals=regime_intervals,
            title=f"Synthetic regimes (WK-means), stride={h2}",
            save_path=os.path.join(output_dir, f"phase1_synth_wk_stride{h2}.png")
        )
        plt.close(wk_fig)

        temporal_fig = plot_historical_coloring(
            prices,
            temporal_labels,
            h1,
            h2,
            regime_intervals=regime_intervals,
            title=f"Synthetic regimes (Temporal OT), stride={h2}",
            save_path=os.path.join(output_dir, f"phase1_synth_temporal_stride{h2}.png")
        )
        plt.close(temporal_fig)


if __name__ == "__main__":
    run_phase1_synthetic()
