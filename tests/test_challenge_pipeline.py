from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antropy")

from rcbd_har.challenge_features import (  # noqa: E402
    LIMB_ORDER,
    extract_multiresolution_features,
    feature_names,
)
from rcbd_har.ul_pairing import (  # noqa: E402
    UL_PAIRS,
    make_grouped_folds,
    select_pair,
    stack_ul_training_variants,
    swap_left_right,
)
from rcbd_har.relational_features import (  # noqa: E402
    extract_relational_features,
    make_ul_relational_variant,
    relational_feature_names,
)


def synthetic_recording(samples: int = 400) -> dict[str, np.ndarray]:
    time = np.arange(samples, dtype=np.float32) / 10.0
    base = np.stack(
        [np.sin(time), np.cos(time * 0.7), np.sin(time * 1.3 + 0.2)], axis=1
    )
    return {limb: base + position for position, limb in enumerate(LIMB_ORDER)}


def test_feature_count_and_temporal_isolation() -> None:
    arrays = synthetic_recording()
    centers = np.asarray([200])
    scales = [1, 2]
    baseline = extract_multiresolution_features(
        arrays, centers, sample_rate=10.0, scales_seconds=scales, batch_size=2
    )
    names = feature_names(scales)
    assert baseline.shape == (1, len(names)) == (1, 648)

    future_changed = {key: value.copy() for key, value in arrays.items()}
    for values in future_changed.values():
        values[200:] *= 3.0
    changed = extract_multiresolution_features(
        future_changed, centers, sample_rate=10.0, scales_seconds=scales, batch_size=2
    )
    past = np.asarray(["__past_" in name for name in names])
    future = np.asarray(["__future_" in name for name in names])
    np.testing.assert_allclose(baseline[:, past], changed[:, past], equal_nan=True)
    assert not np.allclose(baseline[:, future], changed[:, future], equal_nan=True)


def test_ul_pairing_expands_training_rows_in_matching_order() -> None:
    features = np.arange(3 * 20, dtype=np.float32).reshape(3, 20)
    labels = np.asarray([2, 4, 6])
    expanded, expanded_labels = stack_ul_training_variants(features, labels)
    assert expanded.shape == (12, 10)
    np.testing.assert_array_equal(expanded_labels, np.tile(labels, 4))
    for index, pair in enumerate(UL_PAIRS):
        np.testing.assert_array_equal(expanded[index * 3 : (index + 1) * 3], select_pair(features, pair))


def test_grouped_folds_hold_out_every_participant_once() -> None:
    groups = [participant for participant in [f"sbj_{index}" for index in range(18)] for _ in range(4)]
    folds = make_grouped_folds(groups, n_splits=3)
    assert len(folds) == 3
    assert all(len(fold) == 6 for fold in folds)
    assert sorted(value for fold in folds for value in fold) == sorted(set(groups))


def test_left_right_swap_is_an_involution() -> None:
    features = np.arange(3 * 20, dtype=np.float32).reshape(3, 20)
    swapped = swap_left_right(features)
    np.testing.assert_array_equal(swapped[:, :5], features[:, 5:10])
    np.testing.assert_array_equal(swapped[:, 10:15], features[:, 15:20])
    np.testing.assert_array_equal(swap_left_right(swapped), features)


def test_relational_features_are_finite_and_pair_aligned() -> None:
    arrays = synthetic_recording(samples=400)
    centers = np.asarray([200])
    limb, pair = extract_relational_features(
        arrays, centers, sample_rate=10.0, scales_seconds=[1, 2], batch_size=2
    )
    limb_names, pair_names = relational_feature_names([1, 2])
    assert limb.shape == (1, 4, len(limb_names))
    assert pair.shape == (1, 4, len(pair_names))
    assert np.isfinite(limb).all()
    assert np.isfinite(pair).all()
    base = np.arange(1 * 40, dtype=np.float32).reshape(1, 40)
    variant = make_ul_relational_variant(base, limb, pair, pair_index=2)
    assert variant.shape == (1, 20 + 2 * len(limb_names) + len(pair_names))
    pair_only = make_ul_relational_variant(base, limb, pair, pair_index=2, feature_set="pair")
    assert pair_only.shape == (1, 20 + len(pair_names))
    limb_only = make_ul_relational_variant(base, limb, pair, pair_index=2, feature_set="limb")
    assert limb_only.shape == (1, 20 + 2 * len(limb_names))
