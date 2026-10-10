from __future__ import annotations

import numpy as np
import pandas as pd

from rcbd_har.protocol_decoder import protocol_constrained_decode
from rcbd_har.temporal_decoder import viterbi_decode


def _frame(length: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant": ["p0"] * length,
            "recording_id": ["r0"] * length,
            "center": np.arange(length),
        }
    )


def test_zero_bounds_match_standard_viterbi() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.7, 0.3], [0.2, 0.8]])
    transition = np.asarray([[0.9, 0.1], [0.1, 0.9]])
    expected = viterbi_decode(probabilities, _frame(3), transition, 0.5)
    result = protocol_constrained_decode(
        probabilities, _frame(3), transition, 0.5, np.zeros(2, dtype=int)
    )
    np.testing.assert_array_equal(result.predictions, expected)
    assert result.constraints_satisfied


def test_lower_bound_recovers_underrepresented_class() -> None:
    probabilities = np.tile(np.asarray([[0.8, 0.2]]), (12, 1))
    transition = np.asarray([[0.95, 0.05], [0.05, 0.95]])
    result = protocol_constrained_decode(
        probabilities,
        _frame(12),
        transition,
        0.5,
        np.asarray([0, 4]),
    )
    assert result.class_counts[1] >= 4
    assert result.class_biases[1] > 0
    assert result.constraints_satisfied


def test_rejects_misaligned_bounds() -> None:
    probabilities = np.asarray([[0.8, 0.2]])
    transition = np.eye(2)
    try:
        protocol_constrained_decode(
            probabilities, _frame(1), transition, 1.0, np.asarray([0, 1, 2])
        )
    except ValueError as error:
        assert "minimum_counts" in str(error)
    else:
        raise AssertionError("Expected a ValueError")
