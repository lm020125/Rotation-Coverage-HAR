from __future__ import annotations

import numpy as np
import pandas as pd

from rcbd_har.temporal_decoder import (
    estimate_transition_matrix,
    temporal_segments,
    viterbi_decode,
)


def test_temporal_segments_split_recordings_and_gaps() -> None:
    frame = pd.DataFrame(
        {
            "participant": ["p1"] * 5,
            "recording_id": ["a"] * 4 + ["b"],
            "center": [0, 1, 2, 10, 0],
        }
    )
    segments = temporal_segments(frame)
    assert [value.tolist() for value in segments] == [[0, 1, 2], [3], [4]]


def test_transition_matrix_is_row_stochastic() -> None:
    frame = pd.DataFrame({"participant": ["p1"] * 5, "center": [0, 1, 2, 3, 4]})
    transition = estimate_transition_matrix(np.array([0, 0, 0, 1, 1]), frame, 2)
    np.testing.assert_allclose(transition.sum(axis=1), 1.0)
    assert transition[0, 0] > transition[0, 1]
    assert transition[1, 1] > transition[1, 0]


def test_viterbi_uses_persistence_to_remove_isolated_error() -> None:
    frame = pd.DataFrame({"participant": ["p1"] * 3, "center": [0, 1, 2]})
    probabilities = np.array([[0.9, 0.1], [0.45, 0.55], [0.9, 0.1]])
    transition = np.array([[0.99, 0.01], [0.01, 0.99]])
    decoded = viterbi_decode(probabilities, frame, transition, strength=1.0)
    np.testing.assert_array_equal(decoded, [0, 0, 0])


def test_zero_strength_matches_argmax() -> None:
    frame = pd.DataFrame({"participant": ["p1"] * 3, "center": [0, 1, 2]})
    probabilities = np.array([[0.9, 0.1], [0.45, 0.55], [0.2, 0.8]])
    transition = np.array([[0.99, 0.01], [0.01, 0.99]])
    decoded = viterbi_decode(probabilities, frame, transition, strength=0.0)
    np.testing.assert_array_equal(decoded, probabilities.argmax(axis=1))
