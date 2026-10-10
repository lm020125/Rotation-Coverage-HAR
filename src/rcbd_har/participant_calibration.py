from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from rcbd_har.metrics import classification_metrics


AMPLITUDE_FEATURES = ("peak_to_peak", "iqr", "std")


def amplitude_feature_indices(
    names: Sequence[str], feature_kinds: Sequence[str] = AMPLITUDE_FEATURES
) -> np.ndarray:
    """Return indices of positive scale-sensitive feature columns."""
    suffixes = tuple(f"__{kind}" for kind in feature_kinds)
    return np.asarray(
        [index for index, name in enumerate(names) if str(name).endswith(suffixes)],
        dtype=np.int64,
    )


def _positive_medians(values: np.ndarray, indices: np.ndarray, epsilon: float) -> np.ndarray:
    selected = np.asarray(values, dtype=np.float64)[:, indices]
    medians = np.nanmedian(selected, axis=0)
    valid = np.isfinite(medians) & (medians > epsilon)
    return np.where(valid, medians, np.nan)


def fit_amplitude_reference(
    participant_features: Mapping[str, np.ndarray],
    indices: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Fit an equal-participant robust reference without using labels."""
    if not participant_features:
        raise ValueError("At least one participant is required")
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("Amplitude feature indices must be a non-empty vector")
    participant_medians = np.stack(
        [_positive_medians(values, indices, epsilon) for values in participant_features.values()]
    )
    reference = np.nanmedian(participant_medians, axis=0)
    return np.where(np.isfinite(reference) & (reference > epsilon), reference, 1.0)


def calibrate_amplitude_features(
    features: np.ndarray,
    indices: np.ndarray,
    reference: np.ndarray,
    strength: float,
    minimum_factor: float = 0.5,
    maximum_factor: float = 2.0,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Align participant amplitude medians while preserving all other columns."""
    values = np.asarray(features, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    reference = np.asarray(reference, dtype=np.float64)
    if reference.shape != (len(indices),):
        raise ValueError("Reference and amplitude indices must align")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("Calibration strength must be in [0, 1]")
    if not 0.0 < minimum_factor <= maximum_factor:
        raise ValueError("Calibration factor bounds are invalid")
    local = _positive_medians(values, indices, epsilon)
    raw_factor = np.divide(
        reference,
        local,
        out=np.ones_like(reference),
        where=np.isfinite(local) & (local > epsilon),
    )
    raw_factor = np.clip(raw_factor, minimum_factor, maximum_factor)
    factor = np.power(raw_factor, strength)
    calibrated = values.copy()
    calibrated[:, indices] *= factor.astype(np.float32)
    shift_score = float(np.median(np.abs(np.log(raw_factor))))
    return calibrated, shift_score, factor.astype(np.float32)


def compose_null_activity_probabilities(
    activity_probability: np.ndarray,
    conditional_probabilities: np.ndarray,
) -> np.ndarray:
    """Compose P(null) and P(activity class | active) into 19-class probabilities."""
    activity = np.asarray(activity_probability, dtype=np.float64)
    conditional = np.asarray(conditional_probabilities, dtype=np.float64)
    if activity.ndim != 1 or conditional.ndim != 2 or len(activity) != len(conditional):
        raise ValueError("Activity and conditional probabilities must align")
    if conditional.shape[1] == 0:
        raise ValueError("At least one activity class is required")
    activity = np.clip(activity, 0.0, 1.0)
    row_sum = conditional.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0) or not np.isfinite(conditional).all():
        raise ValueError("Conditional probabilities must be finite and have positive mass")
    conditional = conditional / row_sum
    output = np.empty((len(activity), conditional.shape[1] + 1), dtype=np.float32)
    output[:, 0] = 1.0 - activity
    output[:, 1:] = activity[:, None] * conditional
    return output


def apply_null_activity_gate(
    main_predictions: np.ndarray,
    candidate_probabilities: np.ndarray,
    shift_scores: np.ndarray,
    promote_margin: float,
    demote_margin: float,
    minimum_shift: float,
) -> np.ndarray:
    """Only route null/activity disagreements; never rewrite one activity as another."""
    main = np.asarray(main_predictions, dtype=np.int64)
    probabilities = np.asarray(candidate_probabilities, dtype=np.float64)
    shifts = np.asarray(shift_scores, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(main):
        raise ValueError("Candidate probabilities must be [rows, classes]")
    if shifts.shape != (len(main),):
        raise ValueError("Shift scores must have one value per row")
    if not np.isfinite(probabilities).all() or not np.isfinite(shifts).all():
        raise ValueError("Gate inputs must be finite")
    best_activity = probabilities[:, 1:].argmax(axis=1) + 1
    activity_mass = probabilities[np.arange(len(main)), best_activity]
    margin = activity_mass - probabilities[:, 0]
    eligible = shifts >= minimum_shift
    promote = eligible & (main == 0) & (margin >= promote_margin)
    demote = eligible & (main != 0) & (-margin >= demote_margin)
    output = main.copy()
    output[promote] = best_activity[promote]
    output[demote] = 0
    return output


def participant_switch_rates(
    main_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    participant_ids: Sequence[str],
) -> dict[str, float]:
    """Measure how often a routing rule changes each participant."""
    main = np.asarray(main_predictions, dtype=np.int64)
    candidate = np.asarray(candidate_predictions, dtype=np.int64)
    participants = np.asarray(participant_ids, dtype=str)
    if main.shape != candidate.shape or participants.shape != main.shape:
        raise ValueError("Predictions and participant IDs must align")
    return {
        str(participant): float(
            np.mean(candidate[participants == participant] != main[participants == participant])
        )
        for participant in np.unique(participants)
    }


def apply_participant_support_fallback(
    main_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    participant_ids: Sequence[str],
    switch_ceiling: float,
) -> tuple[np.ndarray, dict[str, float], list[str]]:
    """Fall back when a participant's switch rate exceeds inner-OOF support."""
    if not 0.0 <= switch_ceiling <= 1.0:
        raise ValueError("Switch ceiling must be in [0, 1]")
    main = np.asarray(main_predictions, dtype=np.int64)
    candidate = np.asarray(candidate_predictions, dtype=np.int64)
    participants = np.asarray(participant_ids, dtype=str)
    rates = participant_switch_rates(main, candidate, participants)
    unsupported = sorted(
        participant for participant, rate in rates.items() if rate > switch_ceiling + 1e-12
    )
    output = candidate.copy()
    if unsupported:
        output[np.isin(participants, unsupported)] = main[np.isin(participants, unsupported)]
    return output, rates, unsupported


def select_null_activity_rule(
    main_predictions: np.ndarray,
    candidate_probabilities: Mapping[float, np.ndarray],
    shift_scores: Mapping[float, np.ndarray],
    targets: np.ndarray,
    promote_margins: Sequence[float],
    demote_margins: Sequence[float],
    minimum_shifts: Sequence[float],
    num_classes: int,
    minimum_gain: float = 0.0,
) -> tuple[dict[str, float | bool], list[dict[str, float]]]:
    """Select a label-free-at-test rule using only supplied inner-OOF labels."""
    main = np.asarray(main_predictions, dtype=np.int64)
    truth = np.asarray(targets, dtype=np.int64)
    if main.shape != truth.shape:
        raise ValueError("Main predictions and targets must align")
    baseline = classification_metrics(truth, main, num_classes)["macro_f1"]
    scores: list[dict[str, float]] = []
    for strength, probabilities in candidate_probabilities.items():
        if strength not in shift_scores:
            raise ValueError(f"Missing shift scores for strength {strength}")
        for minimum_shift in minimum_shifts:
            for promote_margin in promote_margins:
                for demote_margin in demote_margins:
                    prediction = apply_null_activity_gate(
                        main,
                        probabilities,
                        shift_scores[strength],
                        promote_margin,
                        demote_margin,
                        minimum_shift,
                    )
                    score = classification_metrics(truth, prediction, num_classes)["macro_f1"]
                    scores.append(
                        {
                            "strength": float(strength),
                            "minimum_shift": float(minimum_shift),
                            "promote_margin": float(promote_margin),
                            "demote_margin": float(demote_margin),
                            "inner_center_macro_f1": float(score),
                            "switch_fraction": float(np.mean(prediction != main)),
                        }
                    )
    if not scores:
        raise ValueError("At least one null/activity rule candidate is required")
    best = sorted(
        scores,
        key=lambda value: (
            -value["inner_center_macro_f1"],
            value["switch_fraction"],
            -value["minimum_shift"],
            -value["promote_margin"],
            -value["demote_margin"],
            value["strength"],
        ),
    )[0]
    enabled = best["inner_center_macro_f1"] >= float(baseline) + minimum_gain
    configuration: dict[str, float | bool] = dict(best)
    configuration["enabled"] = bool(enabled)
    configuration["inner_main_center_macro_f1"] = float(baseline)
    return configuration, scores
