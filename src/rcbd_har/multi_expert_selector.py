from __future__ import annotations

from typing import Any

import numpy as np

from rcbd_har.null_refinement import directional_probability_context


def multi_expert_targets(
    main: np.ndarray,
    raw: np.ndarray,
    relational: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Prefer the main model, then raw expert, then relational expert when correct."""

    candidates = [np.asarray(value, dtype=np.int64) for value in (main, raw, relational)]
    truth = np.asarray(targets, dtype=np.int64)
    if any(value.shape != truth.shape for value in candidates) or truth.ndim != 1:
        raise ValueError("Selector predictions and targets must align row-for-row")
    actions = np.zeros(len(truth), dtype=np.int64)
    main_wrong = candidates[0] != truth
    actions[main_wrong & (candidates[1] == truth)] = 1
    actions[main_wrong & (candidates[1] != truth) & (candidates[2] == truth)] = 2
    return actions


def build_multi_expert_features(
    main: np.ndarray,
    raw: np.ndarray,
    relational: np.ndarray,
    raw_score: np.ndarray,
    relational_score: np.ndarray,
    frame: Any,
    num_classes: int,
) -> np.ndarray:
    labels = [np.asarray(value, dtype=np.int64) for value in (main, raw, relational)]
    scores = np.column_stack([raw_score, relational_score]).astype(np.float32)
    if any(value.shape != (len(frame),) for value in labels):
        raise ValueError("Selector labels must align row-for-row")
    if scores.shape != (len(frame), 2) or not np.isfinite(scores).all():
        raise ValueError("Selector scores must be finite and aligned")
    if any(np.any(value < 0) or np.any(value >= num_classes) for value in labels):
        raise ValueError("Selector labels contain an invalid class")
    one_hot = [np.eye(num_classes, dtype=np.float32)[value] for value in labels]
    flags = np.column_stack(
        [
            labels[0] != labels[1],
            labels[0] != labels[2],
            labels[1] != labels[2],
        ]
    ).astype(np.float32)
    temporal: list[np.ndarray] = []
    for radius in (1, 3, 10):
        past, future = directional_probability_context(scores, frame, radius)
        temporal.extend([past, future, scores - past, scores - future])
    output = np.column_stack([scores, *one_hot, flags, *temporal]).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Selector features contain non-finite values")
    return output


def apply_multi_expert_selector(
    main: np.ndarray,
    raw: np.ndarray,
    relational: np.ndarray,
    action_probabilities: np.ndarray,
    margin: float,
) -> np.ndarray:
    if margin < 0.0:
        raise ValueError("Selector margin must be non-negative")
    candidates = np.stack([main, raw, relational]).astype(np.int64)
    probabilities = np.asarray(action_probabilities, dtype=np.float64)
    if probabilities.shape != (candidates.shape[1], 3):
        raise ValueError("Selector probabilities must be [rows, 3]")
    action = probabilities.argmax(axis=1)
    rows = np.arange(candidates.shape[1])
    improvement = probabilities[rows, action] - probabilities[:, 0]
    action[(action != 0) & (improvement < margin)] = 0
    return candidates[action, rows]
