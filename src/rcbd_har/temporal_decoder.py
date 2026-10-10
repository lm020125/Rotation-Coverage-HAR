from __future__ import annotations

from typing import Any

import numpy as np


def temporal_segments(frame: Any) -> list[np.ndarray]:
    """Return center-ordered recording segments without crossing temporal gaps."""
    recording_column = "recording_id" if "recording_id" in frame.columns else "session"
    grouping = ["participant"]
    if recording_column in frame.columns:
        grouping.append(recording_column)
    segments: list[np.ndarray] = []
    for raw_indices in frame.reset_index(drop=True).groupby(grouping, sort=False).indices.values():
        indices = np.asarray(raw_indices, dtype=np.int64)
        centers = frame.iloc[indices].center.to_numpy(np.int64)
        order = np.argsort(centers, kind="stable")
        indices, centers = indices[order], centers[order]
        differences = np.diff(centers)
        if np.any(differences <= 0):
            raise ValueError("Temporal decoding requires unique increasing centers")
        positive = differences[differences > 0]
        expected = float(np.median(positive)) if len(positive) else 1.0
        split_points = np.flatnonzero(differences > 1.5 * expected) + 1
        segments.extend(value for value in np.split(indices, split_points) if len(value))
    return segments


def estimate_transition_matrix(
    targets: np.ndarray,
    frame: Any,
    num_classes: int,
    laplace: float = 1e-2,
) -> np.ndarray:
    """Estimate a row-stochastic class transition matrix within recordings."""
    labels = np.asarray(targets, dtype=np.int64)
    if len(labels) != len(frame):
        raise ValueError("targets and frame must have the same number of rows")
    if laplace <= 0:
        raise ValueError("laplace must be positive")
    counts = np.full((num_classes, num_classes), laplace, dtype=np.float64)
    for segment in temporal_segments(frame):
        sequence = labels[segment]
        if len(sequence) > 1:
            np.add.at(counts, (sequence[:-1], sequence[1:]), 1.0)
    return counts / counts.sum(axis=1, keepdims=True)


def viterbi_decode_segment(
    probabilities: np.ndarray,
    transition_matrix: np.ndarray,
    strength: float,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    transition = np.asarray(transition_matrix, dtype=np.float64)
    if values.ndim != 2 or transition.shape != (values.shape[1], values.shape[1]):
        raise ValueError("probability and transition shapes do not align")
    if strength < 0:
        raise ValueError("strength must be non-negative")
    emission = np.log(np.clip(values, 1e-12, 1.0))
    log_transition = strength * np.log(np.clip(transition, 1e-12, 1.0))
    scores = np.empty_like(emission)
    backpointers = np.zeros(values.shape, dtype=np.int64)
    scores[0] = emission[0]
    for position in range(1, len(values)):
        candidates = scores[position - 1, :, None] + log_transition
        backpointers[position] = candidates.argmax(axis=0)
        scores[position] = emission[position] + candidates.max(axis=0)
    decoded = np.empty(len(values), dtype=np.int64)
    decoded[-1] = scores[-1].argmax()
    for position in range(len(values) - 1, 0, -1):
        decoded[position - 1] = backpointers[position, decoded[position]]
    return decoded


def viterbi_decode(
    probabilities: np.ndarray,
    frame: Any,
    transition_matrix: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Decode each participant recording independently."""
    values = np.asarray(probabilities)
    if len(values) != len(frame):
        raise ValueError("probabilities and frame must have the same number of rows")
    decoded = np.empty(len(values), dtype=np.int64)
    for segment in temporal_segments(frame):
        decoded[segment] = viterbi_decode_segment(
            values[segment], transition_matrix, strength
        )
    return decoded
