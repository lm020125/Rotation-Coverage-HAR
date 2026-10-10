from __future__ import annotations

import numpy as np

from rcbd_har.participant_calibration import (
    amplitude_feature_indices,
    apply_null_activity_gate,
    apply_participant_support_fallback,
    calibrate_amplitude_features,
    compose_null_activity_probabilities,
    fit_amplitude_reference,
    select_null_activity_rule,
)


def test_amplitude_calibration_aligns_only_selected_columns() -> None:
    names = ["a__std", "a__skewness", "a__iqr", "a__peak_to_peak"]
    indices = amplitude_feature_indices(names)
    assert indices.tolist() == [0, 2, 3]
    source = {
        "p0": np.array([[2.0, 7.0, 4.0, 8.0], [2.0, 8.0, 4.0, 8.0]]),
        "p1": np.array([[4.0, 9.0, 8.0, 16.0], [4.0, 10.0, 8.0, 16.0]]),
    }
    reference = fit_amplitude_reference(source, indices)
    target = np.array([[1.5, 11.0, 3.0, 6.0], [1.5, 12.0, 3.0, 6.0]])
    calibrated, shift, factor = calibrate_amplitude_features(
        target, indices, reference, strength=1.0
    )
    np.testing.assert_allclose(np.median(calibrated[:, indices], axis=0), reference)
    np.testing.assert_array_equal(calibrated[:, 1], target[:, 1])
    assert shift > 0
    assert np.all(factor > 1)


def test_null_activity_probabilities_sum_to_one() -> None:
    activity = np.array([0.2, 0.8])
    conditional = np.array([[0.25, 0.75], [0.6, 0.4]])
    probabilities = compose_null_activity_probabilities(activity, conditional)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(probabilities[:, 0], [0.8, 0.2])


def test_gate_never_rewrites_one_activity_as_another() -> None:
    main = np.array([0, 1, 2, 2])
    probabilities = np.array(
        [
            [0.1, 0.2, 0.7],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.8, 0.1, 0.1],
        ]
    )
    selected = apply_null_activity_gate(
        main,
        probabilities,
        np.ones(4),
        promote_margin=0.2,
        demote_margin=0.2,
        minimum_shift=0.0,
    )
    assert selected.tolist() == [2, 0, 2, 0]


def test_participant_support_fallback_reverts_only_unsupported_people() -> None:
    main = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    candidate = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    participants = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    selected, rates, unsupported = apply_participant_support_fallback(
        main, candidate, participants, switch_ceiling=0.5
    )
    assert rates == {"a": 0.25, "b": 0.75}
    assert unsupported == ["b"]
    assert selected.tolist() == [1, 0, 0, 0, 0, 0, 0, 0]


def test_rule_selection_can_fall_back_to_main() -> None:
    main = np.array([0, 1, 2, 0, 1, 2])
    targets = main.copy()
    probabilities = np.full((6, 3), 0.1)
    probabilities[:, 0] = 0.8
    configuration, _ = select_null_activity_rule(
        main,
        {1.0: probabilities},
        {1.0: np.ones(6)},
        targets,
        promote_margins=[0.0],
        demote_margins=[0.0],
        minimum_shifts=[0.0],
        num_classes=3,
        minimum_gain=0.001,
    )
    assert configuration["enabled"] is False
