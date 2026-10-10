from __future__ import annotations

import numpy as np
import pandas as pd

from rcbd_har.protocol_gate import (
    apply_protocol_gate,
    build_protocol_gate_features,
    hard_vote_with_probability_tiebreak,
    protocol_gate_feature_names,
    protocol_gate_training_targets,
    select_protocol_gate_threshold,
)


def test_protocol_gate_features_are_finite_and_label_free() -> None:
    rng = np.random.default_rng(7)
    raw = rng.dirichlet(np.ones(3), size=8)
    aligned = rng.dirichlet(np.ones(3), size=8)
    pairs = rng.dirichlet(np.ones(3), size=(8, 4))
    base = raw.argmax(axis=1)
    protocol = aligned.argmax(axis=1)
    frame = pd.DataFrame(
        {
            "participant": ["p0"] * 8,
            "recording_id": ["r0"] * 8,
            "center": np.arange(8) * 10,
        }
    )

    features, names = build_protocol_gate_features(
        raw,
        aligned,
        pairs,
        base,
        protocol,
        frame,
        temporal_radii=(1, 2),
    )

    assert features.shape == (8, len(protocol_gate_feature_names(3, (1, 2))))
    assert features.shape[1] == len(names)
    assert np.isfinite(features).all()
    assert not any("target" in name or "truth" in name for name in names)


def test_protocol_gate_targets_keep_only_decisive_disagreements() -> None:
    base = np.array([0, 1, 2, 1, 0])
    protocol = np.array([0, 2, 1, 2, 2])
    targets = np.array([0, 1, 1, 0, 3])

    decisive, labels = protocol_gate_training_targets(base, protocol, targets)

    assert decisive.tolist() == [False, True, True, False, False]
    assert labels.tolist() == [1, 0]


def test_apply_protocol_gate_only_changes_disagreements_above_threshold() -> None:
    base = np.array([0, 1, 2, 3])
    protocol = np.array([0, 2, 1, 1])
    probability = np.array([1.0, 0.7, 0.4, 0.5])

    selected = apply_protocol_gate(base, protocol, probability, threshold=0.5)

    assert selected.tolist() == [0, 1, 1, 3]


def test_threshold_selection_uses_inner_predictions() -> None:
    targets = np.array([0, 1, 2, 3])
    base = np.array([0, 1, 0, 3])
    protocol = np.array([1, 2, 2, 0])
    probability = np.array([0.9, 0.8, 0.2, 0.7])

    selected, scores = select_protocol_gate_threshold(
        base,
        protocol,
        probability,
        targets,
        thresholds=[0.5, 0.85],
        num_classes=4,
    )

    assert selected == 0.5
    assert len(scores) == 2
    assert scores[0]["inner_center_macro_f1"] == 1.0


def test_threshold_selection_can_disable_unsupported_gate() -> None:
    targets = np.array([0, 1, 2, 3])
    base = np.array([0, 1, 0, 3])
    protocol = np.array([1, 2, 2, 0])
    probability = np.array([0.9, 0.8, 0.2, 0.7])

    selected, scores = select_protocol_gate_threshold(
        base,
        protocol,
        probability,
        targets,
        thresholds=[0.5, 0.85],
        num_classes=4,
        minimum_gain=1.1,
    )

    assert selected == 1.0
    assert scores[-1]["switch_fraction"] == 0.0


def test_hard_vote_uses_probabilities_only_to_break_ties() -> None:
    predictions = np.array([[0, 1], [0, 1], [0, 2], [1, 2]])
    support = np.full((4, 2, 3), 0.01)
    support[:, 0, 1] = 0.98
    support[:, 1, 2] = 0.98

    voted = hard_vote_with_probability_tiebreak(predictions, support)

    assert voted.tolist() == [0, 2]
