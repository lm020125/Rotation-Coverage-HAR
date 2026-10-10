from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.special import softmax

from rcbd_har.engine import smooth_probabilities


def mean_participant_prior(
    targets: np.ndarray,
    participants_by_row: Sequence[str],
    participants: Sequence[str],
    num_classes: int,
) -> np.ndarray:
    labels = np.asarray(targets, dtype=np.int64)
    groups = np.asarray(participants_by_row, dtype=str)
    priors = []
    for participant in participants:
        selected = groups == str(participant)
        if not np.any(selected):
            raise ValueError(f"Participant {participant} has no rows")
        counts = np.bincount(labels[selected], minlength=num_classes).astype(np.float64)
        priors.append(counts / counts.sum())
    return np.mean(priors, axis=0)


def estimate_null_fraction(
    probabilities: np.ndarray,
    frame: Any,
    mode: str,
    null_index: int = 0,
    smooth_radius: int = 10,
    smooth_sigma: float = 6.0,
) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    if mode == "soft":
        return float(values[:, null_index].mean())
    if mode == "hard":
        smoothed = smooth_probabilities(values, frame, smooth_radius, smooth_sigma)
        return float(np.mean(smoothed.argmax(axis=1) == null_index))
    raise ValueError(f"Unknown Null estimator: {mode}")


def make_shrunk_target_prior(
    source_prior: np.ndarray,
    estimated_null: float,
    rho: float,
    null_index: int = 0,
) -> np.ndarray:
    source = np.asarray(source_prior, dtype=np.float64)
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0,1]")
    target_null = (1 - rho) * source[null_index] + rho * float(estimated_null)
    nonnull = source.copy()
    nonnull[null_index] = 0.0
    nonnull /= nonnull.sum()
    target = (1 - target_null) * nonnull
    target[null_index] = target_null
    return target


def fit_distribution_bias(
    probabilities: np.ndarray,
    target_prior: np.ndarray,
    iterations: int = 150,
    step_size: float = 0.5,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(target_prior, dtype=np.float64)
    if values.ndim != 2 or target.shape != (values.shape[1],):
        raise ValueError("Probability and target-prior shapes do not align")
    if not np.isclose(target.sum(), 1.0) or np.any(target <= 0):
        raise ValueError("target_prior must be positive and sum to one")
    log_values = np.log(np.clip(values, 1e-8, 1.0))
    bias = np.zeros(values.shape[1], dtype=np.float64)
    for _ in range(iterations):
        adjusted_mean = softmax(log_values + bias, axis=1).mean(axis=0)
        delta = np.log(target) - np.log(np.clip(adjusted_mean, 1e-12, 1.0))
        bias += step_size * delta
        bias -= bias.mean()
        if np.max(np.abs(delta)) < 1e-7:
            break
    return bias


def apply_distribution_bias(
    probabilities: np.ndarray, bias: np.ndarray, beta: float
) -> np.ndarray:
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0,1]")
    values = np.asarray(probabilities, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    if bias.shape != (values.shape[1],):
        raise ValueError("bias does not match the class dimension")
    return softmax(np.log(np.clip(values, 1e-8, 1.0)) + beta * bias, axis=1).astype(
        np.float32
    )

