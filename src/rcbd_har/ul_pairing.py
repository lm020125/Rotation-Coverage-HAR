from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold

from rcbd_har.challenge_features import LIMB_ORDER


UL_PAIRS = (
    ("left_wrist", "left_ankle"),
    ("left_wrist", "right_ankle"),
    ("right_wrist", "left_ankle"),
    ("right_wrist", "right_ankle"),
)


def swap_left_right(features: np.ndarray) -> np.ndarray:
    """Swap wrist and ankle feature blocks while preserving feature order within each limb."""
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[1] % len(LIMB_ORDER):
        raise ValueError("features must be [rows, 4 * per_limb_features]")
    blocks = np.split(values, len(LIMB_ORDER), axis=1)
    return np.concatenate([blocks[1], blocks[0], blocks[3], blocks[2]], axis=1)


def make_grouped_folds(
    participants_by_row: Sequence[str], n_splits: int = 3
) -> list[list[str]]:
    groups = np.asarray(participants_by_row, dtype=str)
    if len(np.unique(groups)) < n_splits:
        raise ValueError("Number of participants must be at least n_splits")
    splitter = GroupKFold(n_splits=n_splits)
    return [
        sorted(np.unique(groups[test]).tolist())
        for _, test in splitter.split(np.zeros(len(groups)), groups=groups)
    ]


def pair_feature_indices(n_features: int, pair: tuple[str, str]) -> np.ndarray:
    if n_features % len(LIMB_ORDER):
        raise ValueError(f"Feature count {n_features} is not divisible by four limbs")
    per_limb = n_features // len(LIMB_ORDER)
    indices: list[int] = []
    for limb in pair:
        position = LIMB_ORDER.index(limb)
        indices.extend(range(position * per_limb, (position + 1) * per_limb))
    return np.asarray(indices, dtype=np.int64)


def select_pair(features: np.ndarray, pair: tuple[str, str]) -> np.ndarray:
    values = np.asarray(features)
    return values[:, pair_feature_indices(values.shape[1], pair)]


def stack_ul_training_variants(
    features: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Expand each training center into the four wrist-ankle UL pairings."""
    labels = np.asarray(labels)
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same number of rows")
    return (
        np.concatenate([select_pair(features, pair) for pair in UL_PAIRS], axis=0),
        np.tile(labels, len(UL_PAIRS)),
    )


def average_ul_probabilities(
    predict_proba: Callable[[np.ndarray], np.ndarray], features: np.ndarray
) -> np.ndarray:
    probabilities = [predict_proba(select_pair(features, pair)) for pair in UL_PAIRS]
    shapes = {np.asarray(value).shape for value in probabilities}
    if len(shapes) != 1:
        raise ValueError(f"UL probability shapes differ: {shapes}")
    return np.mean(probabilities, axis=0)
