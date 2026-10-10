from __future__ import annotations

from typing import Any

import numpy as np


def null_candidates(probabilities: np.ndarray, null_index: int = 0) -> np.ndarray:
    values = np.asarray(probabilities)
    if values.ndim != 2 or not 0 <= null_index < values.shape[1]:
        raise ValueError("probabilities must be [rows, classes] with a valid null index")
    masked = values.copy()
    masked[:, null_index] = -np.inf
    return masked.argmax(axis=1).astype(np.int64)


def candidate_sample_weights(
    candidates: np.ndarray, selected: np.ndarray, mode: str
) -> np.ndarray | None:
    if mode == "none":
        return None
    if mode != "candidate_balanced":
        raise ValueError(f"Unknown gate weighting mode: {mode}")
    candidate_values = np.asarray(candidates)[selected]
    counts = np.bincount(candidate_values, minlength=int(candidates.max()) + 1)
    weights = 1.0 / np.maximum(counts[candidate_values], 1)
    return (weights / weights.mean()).astype(np.float32)


def _recording_segments(frame: Any) -> list[np.ndarray]:
    recording_column = "recording_id" if "recording_id" in frame.columns else "session"
    segments: list[np.ndarray] = []
    for indices in frame.reset_index(drop=True).groupby(
        ["participant", recording_column], sort=False
    ).indices.values():
        indices = np.asarray(indices, dtype=np.int64)
        centers = frame.iloc[indices].center.to_numpy(np.int64)
        order = np.argsort(centers, kind="stable")
        indices, centers = indices[order], centers[order]
        differences = np.diff(centers)
        if np.any(differences <= 0):
            raise ValueError("Temporal features require unique increasing centers")
        positive = differences[differences > 0]
        expected = float(np.median(positive)) if len(positive) else 1.0
        split_points = np.flatnonzero(differences > 1.5 * expected) + 1
        segments.extend(np.split(indices, split_points))
    return segments


def directional_probability_context(
    probabilities: np.ndarray, frame: Any, radius_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(probabilities, dtype=np.float32)
    if radius_steps < 0 or len(values) != len(frame):
        raise ValueError("radius_steps must be non-negative and rows must align")
    past = np.empty_like(values)
    future = np.empty_like(values)
    for segment in _recording_segments(frame):
        group = values[segment].astype(np.float64)
        cumulative = np.concatenate(
            [np.zeros((1, group.shape[1]), dtype=np.float64), np.cumsum(group, axis=0)], axis=0
        )
        positions = np.arange(len(group))
        starts = np.maximum(0, positions - radius_steps)
        stops = np.minimum(len(group), positions + radius_steps + 1)
        past[segment] = (
            (cumulative[positions + 1] - cumulative[starts])
            / (positions - starts + 1)[:, None]
        )
        future[segment] = (
            (cumulative[stops] - cumulative[positions])
            / (stops - positions)[:, None]
        )
    return past, future


def null_gate_feature_names(num_classes: int, temporal_radii: tuple[int, ...]) -> list[str]:
    names = [f"log_probability_{index}" for index in range(num_classes)]
    names += [
        "null_probability",
        "candidate_probability",
        "candidate_minus_null",
        "log_candidate_null_odds",
        "probability_entropy",
        "nonnull_top_gap",
        "base_predicts_null",
    ]
    names += [f"candidate_class_{index}" for index in range(1, num_classes)]
    names += [
        f"pair_{source}_{statistic}"
        for source in ("null", "candidate", "candidate_minus_null")
        for statistic in ("mean", "std", "min", "max")
    ]
    names += [
        "pair_null_vote_fraction",
        "pair_candidate_vote_fraction",
        "pair_probability_disagreement",
        "pair_entropy_mean",
        "pair_entropy_std",
    ]
    for radius in temporal_radii:
        names += [
            f"past_{radius}_null",
            f"future_{radius}_null",
            f"past_{radius}_candidate",
            f"future_{radius}_candidate",
            f"past_{radius}_margin",
            f"future_{radius}_margin",
            f"future_minus_past_{radius}_null",
            f"future_minus_past_{radius}_candidate",
        ]
    return names


def build_null_gate_features(
    probabilities: np.ndarray,
    pair_probabilities: np.ndarray,
    frame: Any,
    null_index: int = 0,
    temporal_radii: tuple[int, ...] = (1, 3, 10),
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    values = np.asarray(probabilities, dtype=np.float64)
    pairs = np.asarray(pair_probabilities, dtype=np.float64)
    if values.ndim != 2 or pairs.ndim != 3:
        raise ValueError("Expected probabilities [N,C] and pair probabilities [N,P,C]")
    if pairs.shape[0] != len(values) or pairs.shape[2] != values.shape[1] or len(frame) != len(values):
        raise ValueError("Probability, pair, and frame shapes do not align")
    candidates = null_candidates(values, null_index)
    rows = np.arange(len(values))
    candidate_probability = values[rows, candidates]
    null_probability = values[:, null_index]
    sorted_nonnull = np.sort(np.delete(values, null_index, axis=1), axis=1)
    nonnull_gap = sorted_nonnull[:, -1] - sorted_nonnull[:, -2]
    entropy = -np.sum(values * np.log(values + 1e-12), axis=1)
    basic = np.column_stack(
        [
            null_probability,
            candidate_probability,
            candidate_probability - null_probability,
            np.log((candidate_probability + 1e-8) / (null_probability + 1e-8)),
            entropy,
            nonnull_gap,
            (null_probability >= candidate_probability).astype(np.float64),
        ]
    )
    one_hot = np.zeros((len(values), values.shape[1] - 1), dtype=np.float64)
    one_hot[rows, candidates - (candidates > null_index)] = 1.0

    pair_null = pairs[:, :, null_index]
    pair_candidate = np.take_along_axis(pairs, candidates[:, None, None], axis=2)[:, :, 0]
    pair_margin = pair_candidate - pair_null
    pair_statistics = []
    for source in (pair_null, pair_candidate, pair_margin):
        pair_statistics.extend(
            [
                source.mean(axis=1),
                source.std(axis=1),
                source.min(axis=1),
                source.max(axis=1),
            ]
        )
    pair_votes = pairs.argmax(axis=2)
    pair_entropy = -np.sum(pairs * np.log(pairs + 1e-12), axis=2)
    disagreement = np.mean(np.sum((pairs - values[:, None, :]) ** 2, axis=2), axis=1)
    pair_block = np.column_stack(
        [
            *pair_statistics,
            np.mean(pair_votes == null_index, axis=1),
            np.mean(pair_votes == candidates[:, None], axis=1),
            disagreement,
            pair_entropy.mean(axis=1),
            pair_entropy.std(axis=1),
        ]
    )

    temporal_columns: list[np.ndarray] = []
    for radius in temporal_radii:
        past, future = directional_probability_context(values, frame, radius)
        past_null, future_null = past[:, null_index], future[:, null_index]
        past_candidate = past[rows, candidates]
        future_candidate = future[rows, candidates]
        temporal_columns.extend(
            [
                past_null,
                future_null,
                past_candidate,
                future_candidate,
                past_candidate - past_null,
                future_candidate - future_null,
                future_null - past_null,
                future_candidate - past_candidate,
            ]
        )
    output = np.column_stack(
        [np.log(values + 1e-8), basic, one_hot, pair_block, *temporal_columns]
    ).astype(np.float32)
    names = null_gate_feature_names(values.shape[1], temporal_radii)
    if output.shape[1] != len(names):
        raise RuntimeError(f"Null gate feature mismatch: {output.shape[1]} vs {len(names)}")
    if not np.isfinite(output).all():
        raise ValueError("Null gate features contain non-finite values")
    return output, candidates, names


def refine_null_probabilities(
    probabilities: np.ndarray,
    candidates: np.ndarray,
    gate_active_probability: np.ndarray,
    alpha: float = 0.5,
    null_index: int = 0,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0,1]")
    values = np.asarray(probabilities, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.int64)
    gate = np.asarray(gate_active_probability, dtype=np.float64)
    if len(values) != len(candidates) or len(values) != len(gate):
        raise ValueError("Refinement inputs must have matching rows")
    rows = np.arange(len(values))
    base_null = values[:, null_index]
    base_candidate = values[rows, candidates]
    base_log_odds = np.log((base_candidate + 1e-8) / (base_null + 1e-8))
    gate = np.clip(gate, 1e-6, 1 - 1e-6)
    gate_log_odds = np.log(gate / (1 - gate))
    final_log_odds = (1 - alpha) * base_log_odds + alpha * gate_log_odds
    active = 1.0 / (1.0 + np.exp(-np.clip(final_log_odds, -30, 30)))
    pair_mass = base_null + base_candidate
    output = values.copy()
    output[:, null_index] = pair_mass * (1 - active)
    output[rows, candidates] = pair_mass * active
    return output.astype(np.float32)
