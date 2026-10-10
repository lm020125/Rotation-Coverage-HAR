from __future__ import annotations

import numpy as np
import pandas as pd

from rcbd_har.prior_alignment import (
    apply_distribution_bias,
    estimate_null_fraction,
    fit_distribution_bias,
    make_shrunk_target_prior,
    mean_participant_prior,
)


def test_mean_participant_prior_weights_people_equally() -> None:
    targets = np.asarray([0, 0, 1, 1, 1, 1])
    people = ["A", "A", "B", "B", "B", "B"]
    prior = mean_participant_prior(targets, people, ["A", "B"], num_classes=2)
    np.testing.assert_allclose(prior, [0.5, 0.5])


def test_shrunk_target_preserves_activity_ratios() -> None:
    source = np.asarray([0.4, 0.2, 0.4])
    target = make_shrunk_target_prior(source, estimated_null=0.6, rho=0.5)
    np.testing.assert_allclose(target.sum(), 1.0)
    np.testing.assert_allclose(target[0], 0.5)
    np.testing.assert_allclose(target[1] / target[2], 0.5)


def test_distribution_bias_matches_target_at_full_strength() -> None:
    probabilities = np.asarray(
        [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.6, 0.3, 0.1]], dtype=np.float32
    )
    target = np.asarray([0.4, 0.3, 0.3])
    bias = fit_distribution_bias(probabilities, target, iterations=500)
    aligned = apply_distribution_bias(probabilities, bias, beta=1.0)
    np.testing.assert_allclose(aligned.mean(axis=0), target, atol=1e-5)
    unchanged = apply_distribution_bias(probabilities, bias, beta=0.0)
    np.testing.assert_allclose(unchanged, probabilities, atol=1e-6)


def test_null_estimators_use_only_probabilities() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=np.float32)
    frame = pd.DataFrame(
        {"participant": ["A", "A"], "recording_id": ["r", "r"], "center": [25, 50]}
    )
    assert np.isclose(estimate_null_fraction(probabilities, frame, "soft"), 0.55)
    assert np.isclose(
        estimate_null_fraction(probabilities, frame, "hard", smooth_radius=0), 0.5
    )
