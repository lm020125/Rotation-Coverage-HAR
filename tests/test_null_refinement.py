from __future__ import annotations

import numpy as np
import pandas as pd

from rcbd_har.null_refinement import (
    build_null_gate_features,
    candidate_sample_weights,
    directional_probability_context,
    null_candidates,
    refine_null_probabilities,
)


def probability_fixture() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    probabilities = np.asarray(
        [
            [0.6, 0.3, 0.1],
            [0.4, 0.5, 0.1],
            [0.2, 0.1, 0.7],
            [0.7, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    pairs = np.repeat(probabilities[:, None, :], 4, axis=1)
    pairs[:, 0] += np.asarray([0.01, -0.005, -0.005], dtype=np.float32)
    pairs /= pairs.sum(axis=2, keepdims=True)
    frame = pd.DataFrame(
        {
            "participant": ["A", "A", "B", "B"],
            "recording_id": ["r", "r", "r", "r"],
            "center": [25, 50, 25, 50],
        }
    )
    return probabilities, pairs, frame


def test_null_candidates_choose_highest_nonnull_class() -> None:
    probabilities, _, _ = probability_fixture()
    np.testing.assert_array_equal(null_candidates(probabilities), [1, 1, 2, 1])


def test_temporal_context_does_not_cross_participants() -> None:
    probabilities, _, frame = probability_fixture()
    past, future = directional_probability_context(probabilities, frame, radius_steps=2)
    changed = probabilities.copy()
    changed[2:] = changed[2:, ::-1]
    changed_past, changed_future = directional_probability_context(changed, frame, radius_steps=2)
    np.testing.assert_allclose(past[:2], changed_past[:2])
    np.testing.assert_allclose(future[:2], changed_future[:2])


def test_gate_features_are_finite_and_candidate_conditioned() -> None:
    probabilities, pairs, frame = probability_fixture()
    features, candidates, names = build_null_gate_features(
        probabilities, pairs, frame, temporal_radii=(1,)
    )
    assert features.shape == (4, len(names))
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(candidates, [1, 1, 2, 1])


def test_refinement_preserves_mass_and_alpha_zero_is_identity() -> None:
    probabilities, _, _ = probability_fixture()
    candidates = null_candidates(probabilities)
    unchanged = refine_null_probabilities(probabilities, candidates, np.full(4, 0.9), alpha=0)
    np.testing.assert_allclose(unchanged, probabilities, atol=1e-6)
    refined = refine_null_probabilities(probabilities, candidates, np.full(4, 0.9), alpha=1)
    np.testing.assert_allclose(refined.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(refined[np.arange(4), candidates] > refined[:, 0])


def test_candidate_balanced_weights_equalize_candidate_mass() -> None:
    candidates = np.asarray([1, 1, 1, 2])
    selected = np.asarray([True, True, True, True])
    weights = candidate_sample_weights(candidates, selected, "candidate_balanced")
    assert weights is not None
    np.testing.assert_allclose(weights[candidates == 1].sum(), weights[candidates == 2].sum())
