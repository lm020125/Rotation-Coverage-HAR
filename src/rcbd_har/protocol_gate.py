from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from rcbd_har.metrics import classification_metrics
from rcbd_har.null_refinement import directional_probability_context


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    return -np.sum(values * np.log(values + 1e-12), axis=-1)


def _top_gap(probabilities: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(probabilities, dtype=np.float64), axis=1)
    return ordered[:, -1] - ordered[:, -2]


def protocol_gate_feature_names(
    num_classes: int, temporal_radii: tuple[int, ...]
) -> list[str]:
    names = [f"raw_log_probability_{index}" for index in range(num_classes)]
    names += [f"aligned_log_probability_{index}" for index in range(num_classes)]
    names += [
        "raw_entropy",
        "aligned_entropy",
        "raw_top_gap",
        "aligned_top_gap",
        "raw_max_probability",
        "aligned_max_probability",
        "raw_probability_at_base",
        "raw_probability_at_protocol",
        "aligned_probability_at_base",
        "aligned_probability_at_protocol",
        "raw_base_protocol_log_odds",
        "aligned_base_protocol_log_odds",
        "raw_aligned_js_divergence",
        "base_protocol_disagree",
    ]
    names += [f"base_class_{index}" for index in range(num_classes)]
    names += [f"protocol_class_{index}" for index in range(num_classes)]
    names += [
        "pair_base_probability_mean",
        "pair_base_probability_std",
        "pair_base_probability_min",
        "pair_base_probability_max",
        "pair_protocol_probability_mean",
        "pair_protocol_probability_std",
        "pair_protocol_probability_min",
        "pair_protocol_probability_max",
        "pair_base_minus_protocol_mean",
        "pair_base_minus_protocol_std",
        "pair_base_minus_protocol_min",
        "pair_base_minus_protocol_max",
        "pair_base_vote_fraction",
        "pair_protocol_vote_fraction",
        "pair_vote_agreement",
        "pair_entropy_mean",
        "pair_entropy_std",
        "pair_raw_probability_disagreement",
    ]
    for radius in temporal_radii:
        names += [
            f"raw_past_{radius}_base",
            f"raw_future_{radius}_base",
            f"raw_past_{radius}_protocol",
            f"raw_future_{radius}_protocol",
            f"aligned_past_{radius}_base",
            f"aligned_future_{radius}_base",
            f"aligned_past_{radius}_protocol",
            f"aligned_future_{radius}_protocol",
            f"raw_past_surprise_{radius}",
            f"raw_future_surprise_{radius}",
            f"aligned_past_surprise_{radius}",
            f"aligned_future_surprise_{radius}",
        ]
    return names


def build_protocol_gate_features(
    raw_probabilities: np.ndarray,
    aligned_probabilities: np.ndarray,
    pair_probabilities: np.ndarray,
    base_predictions: np.ndarray,
    protocol_predictions: np.ndarray,
    frame: Any,
    *,
    temporal_radii: tuple[int, ...] = (1, 3, 10),
) -> tuple[np.ndarray, list[str]]:
    """Build label-free evidence for choosing base or protocol predictions."""

    raw = np.asarray(raw_probabilities, dtype=np.float64)
    aligned = np.asarray(aligned_probabilities, dtype=np.float64)
    pairs = np.asarray(pair_probabilities, dtype=np.float64)
    base = np.asarray(base_predictions, dtype=np.int64)
    protocol = np.asarray(protocol_predictions, dtype=np.int64)
    if raw.ndim != 2 or aligned.shape != raw.shape:
        raise ValueError("Raw and aligned probabilities must be matching [rows, classes]")
    if pairs.ndim != 3 or pairs.shape[0] != len(raw) or pairs.shape[2] != raw.shape[1]:
        raise ValueError("Pair probabilities must be [rows, views, classes]")
    if len(frame) != len(raw) or base.shape != (len(raw),) or protocol.shape != (len(raw),):
        raise ValueError("Gate inputs must align row-for-row")
    if np.any(base < 0) or np.any(base >= raw.shape[1]):
        raise ValueError("Base predictions contain an invalid class")
    if np.any(protocol < 0) or np.any(protocol >= raw.shape[1]):
        raise ValueError("Protocol predictions contain an invalid class")
    if not np.isfinite(raw).all() or not np.isfinite(aligned).all() or not np.isfinite(pairs).all():
        raise ValueError("Gate probabilities must be finite")

    rows = np.arange(len(raw))
    raw_base = raw[rows, base]
    raw_protocol = raw[rows, protocol]
    aligned_base = aligned[rows, base]
    aligned_protocol = aligned[rows, protocol]
    mixture = 0.5 * (raw + aligned)
    js_divergence = 0.5 * np.sum(
        raw * np.log((raw + 1e-12) / (mixture + 1e-12))
        + aligned * np.log((aligned + 1e-12) / (mixture + 1e-12)),
        axis=1,
    )
    basic = np.column_stack(
        [
            _entropy(raw),
            _entropy(aligned),
            _top_gap(raw),
            _top_gap(aligned),
            raw.max(axis=1),
            aligned.max(axis=1),
            raw_base,
            raw_protocol,
            aligned_base,
            aligned_protocol,
            np.log((raw_base + 1e-8) / (raw_protocol + 1e-8)),
            np.log((aligned_base + 1e-8) / (aligned_protocol + 1e-8)),
            js_divergence,
            (base != protocol).astype(np.float64),
        ]
    )
    base_one_hot = np.eye(raw.shape[1], dtype=np.float64)[base]
    protocol_one_hot = np.eye(raw.shape[1], dtype=np.float64)[protocol]

    pair_base = np.take_along_axis(pairs, base[:, None, None], axis=2)[:, :, 0]
    pair_protocol = np.take_along_axis(pairs, protocol[:, None, None], axis=2)[:, :, 0]
    pair_margin = pair_base - pair_protocol
    pair_votes = pairs.argmax(axis=2)
    pair_entropy = _entropy(pairs)
    pair_statistics: list[np.ndarray] = []
    for values in (pair_base, pair_protocol, pair_margin):
        pair_statistics.extend(
            [values.mean(1), values.std(1), values.min(1), values.max(1)]
        )
    pair_block = np.column_stack(
        [
            *pair_statistics,
            np.mean(pair_votes == base[:, None], axis=1),
            np.mean(pair_votes == protocol[:, None], axis=1),
            np.max(
                np.stack(
                    [np.mean(pair_votes == index, axis=1) for index in range(raw.shape[1])],
                    axis=1,
                ),
                axis=1,
            ),
            pair_entropy.mean(axis=1),
            pair_entropy.std(axis=1),
            np.mean(np.sum((pairs - raw[:, None, :]) ** 2, axis=2), axis=1),
        ]
    )

    temporal_columns: list[np.ndarray] = []
    for radius in temporal_radii:
        raw_past, raw_future = directional_probability_context(raw, frame, radius)
        aligned_past, aligned_future = directional_probability_context(
            aligned, frame, radius
        )
        temporal_columns.extend(
            [
                raw_past[rows, base],
                raw_future[rows, base],
                raw_past[rows, protocol],
                raw_future[rows, protocol],
                aligned_past[rows, base],
                aligned_future[rows, base],
                aligned_past[rows, protocol],
                aligned_future[rows, protocol],
                np.mean(np.abs(raw - raw_past), axis=1),
                np.mean(np.abs(raw - raw_future), axis=1),
                np.mean(np.abs(aligned - aligned_past), axis=1),
                np.mean(np.abs(aligned - aligned_future), axis=1),
            ]
        )

    output = np.column_stack(
        [
            np.log(raw + 1e-8),
            np.log(aligned + 1e-8),
            basic,
            base_one_hot,
            protocol_one_hot,
            pair_block,
            *temporal_columns,
        ]
    ).astype(np.float32)
    names = protocol_gate_feature_names(raw.shape[1], temporal_radii)
    if output.shape[1] != len(names):
        raise RuntimeError(f"Protocol gate feature mismatch: {output.shape[1]} vs {len(names)}")
    if not np.isfinite(output).all():
        raise ValueError("Protocol gate features contain non-finite values")
    return output, names


def protocol_gate_training_targets(
    base_predictions: np.ndarray,
    protocol_predictions: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return decisive disagreements and 1 when the base prediction is correct."""

    base = np.asarray(base_predictions, dtype=np.int64)
    protocol = np.asarray(protocol_predictions, dtype=np.int64)
    truth = np.asarray(targets, dtype=np.int64)
    if base.shape != protocol.shape or base.shape != truth.shape or base.ndim != 1:
        raise ValueError("Predictions and targets must be aligned one-dimensional arrays")
    base_correct = base == truth
    protocol_correct = protocol == truth
    decisive = (base != protocol) & (base_correct != protocol_correct)
    labels = base_correct[decisive].astype(np.int64)
    return decisive, labels


def apply_protocol_gate(
    base_predictions: np.ndarray,
    protocol_predictions: np.ndarray,
    base_better_probability: np.ndarray,
    threshold: float,
) -> np.ndarray:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Gate threshold must lie in [0, 1]")
    base = np.asarray(base_predictions, dtype=np.int64)
    protocol = np.asarray(protocol_predictions, dtype=np.int64)
    probability = np.asarray(base_better_probability, dtype=np.float64)
    if base.shape != protocol.shape or base.shape != probability.shape or base.ndim != 1:
        raise ValueError("Gate inputs must be aligned one-dimensional arrays")
    output = protocol.copy()
    choose_base = (base != protocol) & (probability >= threshold)
    output[choose_base] = base[choose_base]
    return output


def hard_vote_with_probability_tiebreak(
    predictions: np.ndarray,
    probability_support: np.ndarray,
) -> np.ndarray:
    """Equal hard vote with probabilities used only among tied classes."""

    labels = np.asarray(predictions, dtype=np.int64)
    support = np.asarray(probability_support, dtype=np.float64)
    if labels.ndim != 2 or support.ndim != 3:
        raise ValueError("Expected predictions [models, rows] and support [models, rows, classes]")
    if labels.shape != support.shape[:2]:
        raise ValueError("Prediction and probability model/row axes must match")
    if np.any(labels < 0) or np.any(labels >= support.shape[2]):
        raise ValueError("Predictions contain an invalid class")
    if not np.isfinite(support).all():
        raise ValueError("Probability support must be finite")
    counts = np.eye(support.shape[2], dtype=np.int16)[labels].sum(axis=0)
    tied = counts == counts.max(axis=1, keepdims=True)
    mean_support = support.mean(axis=0)
    return np.where(tied, mean_support, -np.inf).argmax(axis=1).astype(np.int64)


def select_protocol_gate_threshold(
    base_predictions: np.ndarray,
    protocol_predictions: np.ndarray,
    base_better_probability: np.ndarray,
    targets: np.ndarray,
    thresholds: Iterable[float],
    num_classes: int,
    minimum_gain: float = 0.0,
) -> tuple[float, list[dict[str, float]]]:
    candidates = [float(value) for value in thresholds]
    if not candidates or any(value < 0.0 or value > 1.0 for value in candidates):
        raise ValueError("At least one valid gate threshold is required")
    if minimum_gain < 0.0:
        raise ValueError("minimum_gain must be non-negative")
    truth = np.asarray(targets, dtype=np.int64)
    scores: list[dict[str, float]] = []
    for threshold in candidates:
        predictions = apply_protocol_gate(
            base_predictions,
            protocol_predictions,
            base_better_probability,
            threshold,
        )
        score = classification_metrics(truth, predictions, num_classes)["macro_f1"]
        scores.append(
            {
                "threshold": threshold,
                "inner_center_macro_f1": float(score),
                "switch_fraction": float(np.mean(predictions != protocol_predictions)),
            }
        )
    selected = sorted(
        scores,
        key=lambda value: (
            -value["inner_center_macro_f1"],
            abs(value["threshold"] - 0.5),
            -value["threshold"],
        ),
    )[0]
    protocol_score = classification_metrics(
        truth, np.asarray(protocol_predictions, dtype=np.int64), num_classes
    )["macro_f1"]
    if selected["inner_center_macro_f1"] < protocol_score + minimum_gain:
        scores.append(
            {
                "threshold": 1.0,
                "inner_center_macro_f1": float(protocol_score),
                "switch_fraction": 0.0,
            }
        )
        return 1.0, scores
    return float(selected["threshold"]), scores
