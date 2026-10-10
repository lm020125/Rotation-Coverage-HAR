from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from rcbd_har.metrics import classification_metrics
from rcbd_har.null_refinement import directional_probability_context
from rcbd_har.temporal_decoder import temporal_segments


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    return -np.sum(values * np.log(values + 1e-12), axis=1)


def _top_gap(probabilities: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(probabilities, dtype=np.float64), axis=1)
    return ordered[:, -1] - ordered[:, -2]


def residual_expert_feature_names(
    num_classes: int,
    temporal_radii: tuple[int, ...],
    *,
    disagreement_aware: bool = False,
    auxiliary_aware: bool = False,
) -> list[str]:
    names = [f"main_log_probability_{index}" for index in range(num_classes)]
    names += [f"expert_log_probability_{index}" for index in range(num_classes)]
    names += [f"expert_minus_main_{index}" for index in range(num_classes)]
    names += [
        "main_entropy",
        "expert_entropy",
        "main_top_gap",
        "expert_top_gap",
        "main_max_probability",
        "expert_max_probability",
        "main_probability_at_main",
        "main_probability_at_expert",
        "expert_probability_at_main",
        "expert_probability_at_expert",
        "main_prediction_log_odds",
        "expert_prediction_log_odds",
        "main_expert_js_divergence",
        "main_expert_disagree",
    ]
    names += [f"main_class_{index}" for index in range(num_classes)]
    names += [f"expert_class_{index}" for index in range(num_classes)]
    if disagreement_aware:
        names += [f"member_probability_std_{index}" for index in range(num_classes)]
        names += [f"member_vote_fraction_{index}" for index in range(num_classes)]
        names += [
            "main_class_member_probability_std",
            "expert_class_member_probability_mean",
            "expert_class_member_probability_std",
            "main_vote_fraction",
            "expert_vote_fraction",
            "top_vote_fraction",
            "top_vote_gap",
            "pairwise_prediction_disagreement",
            "hard_soft_prediction_disagree",
            "member_entropy_mean",
            "member_entropy_std",
            "member_probability_dispersion_mean",
            "member_probability_dispersion_max",
        ]
    if auxiliary_aware:
        names += [f"auxiliary_log_probability_{index}" for index in range(num_classes)]
        names += [f"auxiliary_class_{index}" for index in range(num_classes)]
        names += [
            "auxiliary_entropy",
            "auxiliary_top_gap",
            "auxiliary_max_probability",
            "auxiliary_probability_at_main",
            "auxiliary_probability_at_expert",
            "auxiliary_main_log_odds",
            "auxiliary_expert_agreement",
            "auxiliary_main_agreement",
            "auxiliary_main_js_divergence",
            "auxiliary_expert_js_divergence",
        ]
    for radius in temporal_radii:
        names += [
            f"main_past_{radius}_main",
            f"main_future_{radius}_main",
            f"main_past_{radius}_expert",
            f"main_future_{radius}_expert",
            f"expert_past_{radius}_main",
            f"expert_future_{radius}_main",
            f"expert_past_{radius}_expert",
            f"expert_future_{radius}_expert",
            f"main_past_surprise_{radius}",
            f"main_future_surprise_{radius}",
            f"expert_past_surprise_{radius}",
            f"expert_future_surprise_{radius}",
        ]
    return names


def build_residual_expert_features(
    main_probabilities: np.ndarray,
    expert_probabilities: np.ndarray,
    main_predictions: np.ndarray,
    expert_predictions: np.ndarray,
    frame: Any,
    *,
    temporal_radii: tuple[int, ...] = (1, 3, 10),
    member_probabilities: np.ndarray | None = None,
    member_predictions: np.ndarray | None = None,
    auxiliary_probabilities: np.ndarray | None = None,
    auxiliary_predictions: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build label-free evidence for routing between the main model and expert."""

    main = np.asarray(main_probabilities, dtype=np.float64)
    expert = np.asarray(expert_probabilities, dtype=np.float64)
    main_labels = np.asarray(main_predictions, dtype=np.int64)
    expert_labels = np.asarray(expert_predictions, dtype=np.int64)
    if main.ndim != 2 or expert.shape != main.shape:
        raise ValueError("Main and expert probabilities must be matching [rows, classes]")
    if len(frame) != len(main):
        raise ValueError("The feature frame must align row-for-row")
    if main_labels.shape != (len(main),) or expert_labels.shape != (len(main),):
        raise ValueError("Main and expert predictions must align row-for-row")
    if not np.isfinite(main).all() or not np.isfinite(expert).all():
        raise ValueError("Routing probabilities must be finite")
    if np.any(main_labels < 0) or np.any(main_labels >= main.shape[1]):
        raise ValueError("Main predictions contain an invalid class")
    if np.any(expert_labels < 0) or np.any(expert_labels >= main.shape[1]):
        raise ValueError("Expert predictions contain an invalid class")

    rows = np.arange(len(main))
    main_at_main = main[rows, main_labels]
    main_at_expert = main[rows, expert_labels]
    expert_at_main = expert[rows, main_labels]
    expert_at_expert = expert[rows, expert_labels]
    mixture = 0.5 * (main + expert)
    js_divergence = 0.5 * np.sum(
        main * np.log((main + 1e-12) / (mixture + 1e-12))
        + expert * np.log((expert + 1e-12) / (mixture + 1e-12)),
        axis=1,
    )
    basic = np.column_stack(
        [
            _entropy(main),
            _entropy(expert),
            _top_gap(main),
            _top_gap(expert),
            main.max(axis=1),
            expert.max(axis=1),
            main_at_main,
            main_at_expert,
            expert_at_main,
            expert_at_expert,
            np.log((main_at_main + 1e-8) / (main_at_expert + 1e-8)),
            np.log((expert_at_main + 1e-8) / (expert_at_expert + 1e-8)),
            js_divergence,
            (main_labels != expert_labels).astype(np.float64),
        ]
    )
    main_one_hot = np.eye(main.shape[1], dtype=np.float64)[main_labels]
    expert_one_hot = np.eye(main.shape[1], dtype=np.float64)[expert_labels]

    disagreement_columns: list[np.ndarray] = []
    disagreement_aware = member_probabilities is not None or member_predictions is not None
    if disagreement_aware:
        if member_probabilities is None or member_predictions is None:
            raise ValueError("Member probabilities and predictions must be supplied together")
        member_values = np.asarray(member_probabilities, dtype=np.float64)
        member_labels = np.asarray(member_predictions, dtype=np.int64)
        if member_values.ndim != 3 or member_values.shape[1:] != main.shape:
            raise ValueError("Member probabilities must be [models, rows, classes]")
        if member_labels.shape != member_values.shape[:2]:
            raise ValueError("Member predictions must be [models, rows]")
        if not np.isfinite(member_values).all():
            raise ValueError("Member probabilities must be finite")
        if np.any(member_labels < 0) or np.any(member_labels >= main.shape[1]):
            raise ValueError("Member predictions contain an invalid class")
        model_count = member_values.shape[0]
        vote_counts = np.eye(main.shape[1], dtype=np.float64)[member_labels].sum(axis=0)
        vote_fraction = vote_counts / model_count
        ordered_votes = np.sort(vote_fraction, axis=1)
        member_std = member_values.std(axis=0)
        main_member = np.take_along_axis(
            member_values, main_labels[None, :, None], axis=2
        )[:, :, 0]
        expert_member = np.take_along_axis(
            member_values, expert_labels[None, :, None], axis=2
        )[:, :, 0]
        main_votes = vote_fraction[rows, main_labels]
        expert_votes = vote_fraction[rows, expert_labels]
        member_entropy = -np.sum(
            member_values * np.log(member_values + 1e-12), axis=2
        )
        if model_count > 1:
            agreements = np.zeros(len(main), dtype=np.float64)
            for first in range(model_count):
                for second in range(first + 1, model_count):
                    agreements += member_labels[first] == member_labels[second]
            pair_count = model_count * (model_count - 1) / 2
            pairwise_disagreement = 1.0 - agreements / pair_count
        else:
            pairwise_disagreement = np.zeros(len(main), dtype=np.float64)
        disagreement_columns = [
            member_std,
            vote_fraction,
            main_member.std(axis=0),
            expert_member.mean(axis=0),
            expert_member.std(axis=0),
            main_votes,
            expert_votes,
            ordered_votes[:, -1],
            ordered_votes[:, -1] - ordered_votes[:, -2],
            pairwise_disagreement,
            (main.argmax(axis=1) != main_labels).astype(np.float64),
            member_entropy.mean(axis=0),
            member_entropy.std(axis=0),
            member_std.mean(axis=1),
            member_std.max(axis=1),
        ]

    auxiliary_columns: list[np.ndarray] = []
    auxiliary_aware = auxiliary_probabilities is not None or auxiliary_predictions is not None
    if auxiliary_aware:
        if auxiliary_probabilities is None or auxiliary_predictions is None:
            raise ValueError("Auxiliary probabilities and predictions must be supplied together")
        auxiliary = np.asarray(auxiliary_probabilities, dtype=np.float64)
        auxiliary_labels = np.asarray(auxiliary_predictions, dtype=np.int64)
        if auxiliary.shape != main.shape or auxiliary_labels.shape != main_labels.shape:
            raise ValueError("Auxiliary evidence must align row-for-row")
        if not np.isfinite(auxiliary).all():
            raise ValueError("Auxiliary probabilities must be finite")
        if np.any(auxiliary_labels < 0) or np.any(auxiliary_labels >= main.shape[1]):
            raise ValueError("Auxiliary predictions contain an invalid class")
        auxiliary_main = auxiliary[rows, main_labels]
        auxiliary_expert = auxiliary[rows, expert_labels]
        auxiliary_main_mixture = 0.5 * (auxiliary + main)
        auxiliary_expert_mixture = 0.5 * (auxiliary + expert)
        auxiliary_columns = [
            np.log(auxiliary + 1e-8),
            np.eye(main.shape[1], dtype=np.float64)[auxiliary_labels],
            _entropy(auxiliary),
            _top_gap(auxiliary),
            auxiliary.max(axis=1),
            auxiliary_main,
            auxiliary_expert,
            np.log((auxiliary_main + 1e-8) / (auxiliary_expert + 1e-8)),
            (auxiliary_labels == expert_labels).astype(np.float64),
            (auxiliary_labels == main_labels).astype(np.float64),
            0.5
            * np.sum(
                auxiliary
                * np.log((auxiliary + 1e-12) / (auxiliary_main_mixture + 1e-12))
                + main
                * np.log((main + 1e-12) / (auxiliary_main_mixture + 1e-12)),
                axis=1,
            ),
            0.5
            * np.sum(
                auxiliary
                * np.log((auxiliary + 1e-12) / (auxiliary_expert_mixture + 1e-12))
                + expert
                * np.log((expert + 1e-12) / (auxiliary_expert_mixture + 1e-12)),
                axis=1,
            ),
        ]

    temporal_columns: list[np.ndarray] = []
    for radius in temporal_radii:
        main_past, main_future = directional_probability_context(main, frame, radius)
        expert_past, expert_future = directional_probability_context(
            expert, frame, radius
        )
        temporal_columns.extend(
            [
                main_past[rows, main_labels],
                main_future[rows, main_labels],
                main_past[rows, expert_labels],
                main_future[rows, expert_labels],
                expert_past[rows, main_labels],
                expert_future[rows, main_labels],
                expert_past[rows, expert_labels],
                expert_future[rows, expert_labels],
                np.mean(np.abs(main - main_past), axis=1),
                np.mean(np.abs(main - main_future), axis=1),
                np.mean(np.abs(expert - expert_past), axis=1),
                np.mean(np.abs(expert - expert_future), axis=1),
            ]
        )

    output = np.column_stack(
        [
            np.log(main + 1e-8),
            np.log(expert + 1e-8),
            expert - main,
            basic,
            main_one_hot,
            expert_one_hot,
            *disagreement_columns,
            *auxiliary_columns,
            *temporal_columns,
        ]
    ).astype(np.float32)
    names = residual_expert_feature_names(
        main.shape[1],
        temporal_radii,
        disagreement_aware=disagreement_aware,
        auxiliary_aware=auxiliary_aware,
    )
    if output.shape[1] != len(names):
        raise RuntimeError(f"Residual expert feature mismatch: {output.shape[1]} vs {len(names)}")
    if not np.isfinite(output).all():
        raise ValueError("Residual expert features contain non-finite values")
    return output, names


def residual_expert_training_targets(
    main_predictions: np.ndarray,
    expert_predictions: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return decisive disagreements and 1 exactly when the main model is correct."""

    main = np.asarray(main_predictions, dtype=np.int64)
    expert = np.asarray(expert_predictions, dtype=np.int64)
    truth = np.asarray(targets, dtype=np.int64)
    if main.shape != expert.shape or main.shape != truth.shape or main.ndim != 1:
        raise ValueError("Predictions and targets must be aligned one-dimensional arrays")
    main_correct = main == truth
    expert_correct = expert == truth
    decisive = (main != expert) & (main_correct != expert_correct)
    return decisive, main_correct[decisive].astype(np.int64)


def residual_route_scope(
    main_predictions: np.ndarray,
    expert_predictions: np.ndarray,
    *,
    scope: str,
    focus_classes: Iterable[int] = (),
) -> np.ndarray:
    """Return label-free rows where a residual expert is allowed to intervene."""

    main = np.asarray(main_predictions, dtype=np.int64)
    expert = np.asarray(expert_predictions, dtype=np.int64)
    if main.shape != expert.shape or main.ndim != 1:
        raise ValueError("Routing scope predictions must align row-for-row")
    if scope == "all":
        return np.ones(len(main), dtype=bool)
    if scope != "focus":
        raise ValueError(f"Unknown residual routing scope: {scope}")
    classes = np.asarray(sorted(set(int(value) for value in focus_classes)), dtype=np.int64)
    if not len(classes):
        raise ValueError("Focus routing requires at least one focus class")
    return np.isin(main, classes) | np.isin(expert, classes)


def residual_utility_training_targets(
    main_predictions: np.ndarray,
    expert_predictions: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Label disagreements as neutral=0, switching loss=1, or switching gain=2."""

    main = np.asarray(main_predictions, dtype=np.int64)
    expert = np.asarray(expert_predictions, dtype=np.int64)
    truth = np.asarray(targets, dtype=np.int64)
    if main.shape != expert.shape or main.shape != truth.shape or main.ndim != 1:
        raise ValueError("Predictions and targets must be aligned one-dimensional arrays")
    disagreements = main != expert
    labels = np.zeros(int(disagreements.sum()), dtype=np.int64)
    selected_main = main[disagreements]
    selected_expert = expert[disagreements]
    selected_truth = truth[disagreements]
    labels[selected_main == selected_truth] = 1
    labels[selected_expert == selected_truth] = 2
    return disagreements, labels


def fit_pair_expert_logit_offsets(
    main_predictions: np.ndarray,
    expert_predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    shrinkage: float,
) -> np.ndarray:
    """Estimate shrunk expert-win prior offsets for directed confusion pairs."""

    if num_classes <= 1 or shrinkage <= 0.0:
        raise ValueError("Pair calibration requires valid classes and positive shrinkage")
    main = np.asarray(main_predictions, dtype=np.int64)
    expert = np.asarray(expert_predictions, dtype=np.int64)
    truth = np.asarray(targets, dtype=np.int64)
    if main.shape != expert.shape or main.shape != truth.shape or main.ndim != 1:
        raise ValueError("Pair calibration inputs must align row-for-row")
    if np.any(main < 0) or np.any(main >= num_classes):
        raise ValueError("Main predictions contain an invalid class")
    if np.any(expert < 0) or np.any(expert >= num_classes):
        raise ValueError("Expert predictions contain an invalid class")
    main_correct = main == truth
    expert_correct = expert == truth
    decisive = (main != expert) & (main_correct != expert_correct)
    pair_ids = main * num_classes + expert
    pair_count = num_classes * num_classes
    expert_wins = np.bincount(
        pair_ids[decisive & expert_correct], minlength=pair_count
    ).astype(np.float64)
    main_wins = np.bincount(
        pair_ids[decisive & main_correct], minlength=pair_count
    ).astype(np.float64)
    total_expert = expert_wins.sum()
    total_decisive = total_expert + main_wins.sum()
    global_expert_prior = (total_expert + 1.0) / (total_decisive + 2.0)
    pair_prior = (expert_wins + shrinkage * global_expert_prior) / (
        expert_wins + main_wins + shrinkage
    )
    epsilon = 1e-6
    global_logit = np.log(
        np.clip(global_expert_prior, epsilon, 1.0 - epsilon)
        / np.clip(1.0 - global_expert_prior, epsilon, 1.0 - epsilon)
    )
    pair_prior = np.clip(pair_prior, epsilon, 1.0 - epsilon)
    return (np.log(pair_prior / (1.0 - pair_prior)) - global_logit).astype(
        np.float32
    )


def apply_pair_expert_logit_offsets(
    main_better_probability: np.ndarray,
    main_predictions: np.ndarray,
    expert_predictions: np.ndarray,
    offsets: np.ndarray,
    strength: float,
    num_classes: int,
) -> np.ndarray:
    """Prior-correct a gate score using directed-pair expert-win evidence."""

    if strength < 0.0:
        raise ValueError("Pair calibration strength must be non-negative")
    probability = np.asarray(main_better_probability, dtype=np.float64)
    main = np.asarray(main_predictions, dtype=np.int64)
    expert = np.asarray(expert_predictions, dtype=np.int64)
    pair_offsets = np.asarray(offsets, dtype=np.float64)
    if probability.shape != main.shape or main.shape != expert.shape or main.ndim != 1:
        raise ValueError("Pair calibration predictions must align row-for-row")
    if pair_offsets.shape != (num_classes * num_classes,):
        raise ValueError("Pair offsets do not match the number of classes")
    if not np.isfinite(probability).all() or not np.isfinite(pair_offsets).all():
        raise ValueError("Pair calibration inputs must be finite")
    epsilon = 1e-6
    expert_probability = np.clip(1.0 - probability, epsilon, 1.0 - epsilon)
    expert_logit = np.log(expert_probability / (1.0 - expert_probability))
    pair_ids = main * num_classes + expert
    adjusted_expert = 1.0 / (
        1.0 + np.exp(-(expert_logit + strength * pair_offsets[pair_ids]))
    )
    return (1.0 - adjusted_expert).astype(np.float32)


def apply_residual_expert_gate(
    main_predictions: np.ndarray,
    expert_predictions: np.ndarray,
    main_better_probability: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Use the expert only below the learned main-trust threshold."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Gate threshold must lie in [0, 1]")
    main = np.asarray(main_predictions, dtype=np.int64)
    expert = np.asarray(expert_predictions, dtype=np.int64)
    probability = np.asarray(main_better_probability, dtype=np.float64)
    if main.shape != expert.shape or main.shape != probability.shape or main.ndim != 1:
        raise ValueError("Gate inputs must be aligned one-dimensional arrays")
    output = main.copy()
    choose_expert = (main != expert) & (probability < threshold)
    output[choose_expert] = expert[choose_expert]
    return output


def conservative_residual_majority(
    main_predictions: np.ndarray,
    expert_predictions: np.ndarray,
    member_predictions: np.ndarray,
) -> np.ndarray:
    """Call the shared expert only when a strict majority of gates switches."""

    main = np.asarray(main_predictions, dtype=np.int64)
    expert = np.asarray(expert_predictions, dtype=np.int64)
    members = np.asarray(member_predictions, dtype=np.int64)
    if main.shape != expert.shape or main.ndim != 1:
        raise ValueError("Residual ensemble main and expert predictions must align")
    if members.ndim != 2 or members.shape[1] != len(main) or not len(members):
        raise ValueError("Residual ensemble members must be [models, rows]")
    valid = (members == main[None, :]) | (members == expert[None, :])
    if not valid.all():
        raise ValueError("Residual ensemble members must choose the shared main or expert")
    expert_votes = np.sum((members == expert[None, :]) & (main != expert)[None, :], axis=0)
    output = main.copy()
    strict_majority = expert_votes > len(members) / 2
    output[strict_majority] = expert[strict_majority]
    return output


def aggregate_pair_segment_scores(
    main_predictions: np.ndarray,
    expert_predictions: np.ndarray,
    main_better_probability: np.ndarray,
    frame: Any,
    *,
    blend: float,
    minimum_length: int,
) -> np.ndarray:
    """Shrink window scores toward their contiguous directed-pair segment mean.

    Segments use only recording boundaries plus the label-free ``(main, expert)``
    prediction pair. ``blend=1`` is the unchanged window-level gate, while
    ``blend=0`` assigns one mean score to the full pair-consistent segment.
    """

    if not 0.0 <= blend <= 1.0:
        raise ValueError("Segment blend must lie in [0, 1]")
    if minimum_length < 1:
        raise ValueError("Segment minimum length must be positive")
    main = np.asarray(main_predictions, dtype=np.int64)
    expert = np.asarray(expert_predictions, dtype=np.int64)
    probability = np.asarray(main_better_probability, dtype=np.float64)
    if main.shape != expert.shape or main.shape != probability.shape or main.ndim != 1:
        raise ValueError("Segment gate inputs must be aligned one-dimensional arrays")
    if len(frame) != len(main) or not np.isfinite(probability).all():
        raise ValueError("Segment gate frame and probabilities must align and be finite")
    if blend == 1.0:
        return probability.astype(np.float32, copy=True)

    output = probability.copy()
    for recording in temporal_segments(frame):
        pair_changes = (main[recording][1:] != main[recording][:-1]) | (
            expert[recording][1:] != expert[recording][:-1]
        )
        for run in np.split(recording, np.flatnonzero(pair_changes) + 1):
            if len(run) < minimum_length:
                continue
            segment_mean = float(probability[run].mean())
            output[run] = blend * probability[run] + (1.0 - blend) * segment_mean
    return output.astype(np.float32)


def apply_residual_utility_gate(
    main_predictions: np.ndarray,
    expert_predictions: np.ndarray,
    utility_probabilities: np.ndarray,
    loss_cost: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Switch when P(gain) - loss_cost * P(loss) exceeds the threshold."""

    if loss_cost < 0.0:
        raise ValueError("The switching loss cost must be non-negative")
    probabilities = np.asarray(utility_probabilities, dtype=np.float64)
    main = np.asarray(main_predictions, dtype=np.int64)
    expert = np.asarray(expert_predictions, dtype=np.int64)
    if probabilities.shape != (len(main), 3):
        raise ValueError("Utility probabilities must be [rows, 3]")
    if main.shape != expert.shape or main.ndim != 1:
        raise ValueError("Utility gate predictions must align row-for-row")
    if not np.isfinite(probabilities).all():
        raise ValueError("Utility probabilities must be finite")
    utility = probabilities[:, 2] - loss_cost * probabilities[:, 1]
    output = main.copy()
    selected = (main != expert) & (utility > threshold)
    output[selected] = expert[selected]
    return output, utility.astype(np.float32)


def select_residual_utility_rule(
    main_predictions: np.ndarray,
    expert_predictions: np.ndarray,
    utility_probabilities: np.ndarray,
    targets: np.ndarray,
    loss_costs: Iterable[float],
    thresholds: Iterable[float],
    num_classes: int,
    minimum_gain: float = 0.0,
) -> tuple[float, float, list[dict[str, float]]]:
    """Select a utility rule on inner OOF and otherwise retain the main model."""

    costs = [float(value) for value in loss_costs]
    candidates = [float(value) for value in thresholds]
    if not costs or any(value < 0.0 for value in costs):
        raise ValueError("At least one non-negative switching loss cost is required")
    if not candidates:
        raise ValueError("At least one utility threshold is required")
    truth = np.asarray(targets, dtype=np.int64)
    scores: list[dict[str, float]] = []
    for loss_cost in costs:
        for threshold in candidates:
            prediction, _ = apply_residual_utility_gate(
                main_predictions,
                expert_predictions,
                utility_probabilities,
                loss_cost,
                threshold,
            )
            score = classification_metrics(truth, prediction, num_classes)["macro_f1"]
            scores.append(
                {
                    "loss_cost": loss_cost,
                    "threshold": threshold,
                    "inner_center_macro_f1": float(score),
                    "expert_switch_fraction": float(
                        np.mean(prediction != np.asarray(main_predictions))
                    ),
                }
            )
    selected = sorted(
        scores,
        key=lambda value: (
            -value["inner_center_macro_f1"],
            abs(value["loss_cost"] - 1.0),
            abs(value["threshold"]),
        ),
    )[0]
    main_score = classification_metrics(
        truth, np.asarray(main_predictions, dtype=np.int64), num_classes
    )["macro_f1"]
    if selected["inner_center_macro_f1"] < main_score + minimum_gain:
        scores.append(
            {
                "loss_cost": 1.0,
                "threshold": 2.0,
                "inner_center_macro_f1": float(main_score),
                "expert_switch_fraction": 0.0,
            }
        )
        return 1.0, 2.0, scores
    return float(selected["loss_cost"]), float(selected["threshold"]), scores


def select_residual_expert_threshold(
    main_predictions: np.ndarray,
    expert_predictions: np.ndarray,
    main_better_probability: np.ndarray,
    targets: np.ndarray,
    thresholds: Iterable[float],
    num_classes: int,
    minimum_gain: float = 0.0,
) -> tuple[float, list[dict[str, float]]]:
    """Select on inner OOF only and fall back to the unchanged main model."""

    candidates = [float(value) for value in thresholds]
    if not candidates or any(value < 0.0 or value > 1.0 for value in candidates):
        raise ValueError("At least one valid gate threshold is required")
    if minimum_gain < 0.0:
        raise ValueError("minimum_gain must be non-negative")
    truth = np.asarray(targets, dtype=np.int64)
    scores: list[dict[str, float]] = []
    for threshold in candidates:
        prediction = apply_residual_expert_gate(
            main_predictions, expert_predictions, main_better_probability, threshold
        )
        score = classification_metrics(truth, prediction, num_classes)["macro_f1"]
        scores.append(
            {
                "threshold": threshold,
                "inner_center_macro_f1": float(score),
                "expert_switch_fraction": float(
                    np.mean(prediction != np.asarray(main_predictions))
                ),
            }
        )
    selected = sorted(
        scores,
        key=lambda value: (
            -value["inner_center_macro_f1"],
            abs(value["threshold"] - 0.5),
            value["threshold"],
        ),
    )[0]
    main_score = classification_metrics(
        truth, np.asarray(main_predictions, dtype=np.int64), num_classes
    )["macro_f1"]
    if selected["inner_center_macro_f1"] < main_score + minimum_gain:
        scores.append(
            {
                "threshold": 0.0,
                "inner_center_macro_f1": float(main_score),
                "expert_switch_fraction": 0.0,
            }
        )
        return 0.0, scores
    return float(selected["threshold"]), scores
