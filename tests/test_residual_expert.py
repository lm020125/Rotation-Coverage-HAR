import numpy as np
import pandas as pd

from rcbd_har.residual_expert import (
    aggregate_pair_segment_scores,
    apply_residual_expert_gate,
    apply_residual_utility_gate,
    apply_pair_expert_logit_offsets,
    build_residual_expert_features,
    conservative_residual_majority,
    fit_pair_expert_logit_offsets,
    residual_expert_training_targets,
    residual_route_scope,
    residual_utility_training_targets,
    select_residual_expert_threshold,
    select_residual_utility_rule,
)


def test_pair_segment_scores_are_recording_and_pair_consistent() -> None:
    frame = pd.DataFrame(
        {
            "participant": ["p1"] * 6,
            "recording_id": ["r1"] * 6,
            "center": np.arange(6),
        }
    )
    main = np.array([1, 1, 1, 1, 1, 1])
    expert = np.array([2, 2, 2, 3, 3, 3])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    aggregated = aggregate_pair_segment_scores(
        main, expert, scores, frame, blend=0.0, minimum_length=3
    )
    np.testing.assert_allclose(aggregated[:3], 0.2)
    np.testing.assert_allclose(aggregated[3:], 0.8)


def test_pair_segment_blend_one_is_identity() -> None:
    frame = pd.DataFrame(
        {"participant": ["p1"] * 3, "session": ["s1"] * 3, "center": [0, 1, 2]}
    )
    scores = np.array([0.1, 0.5, 0.9])
    output = aggregate_pair_segment_scores(
        np.array([1, 1, 1]),
        np.array([2, 2, 2]),
        scores,
        frame,
        blend=1.0,
        minimum_length=1,
    )
    np.testing.assert_allclose(output, scores)


def test_focus_route_scope_is_label_free_and_symmetric() -> None:
    eligible = residual_route_scope(
        np.array([1, 17, 2, 3]),
        np.array([17, 4, 5, 18]),
        scope="focus",
        focus_classes=(17, 18),
    )
    np.testing.assert_array_equal(eligible, [True, True, False, True])


def test_conservative_residual_majority_retains_main_on_tie() -> None:
    main = np.array([1, 1, 1])
    expert = np.array([2, 2, 2])
    members = np.array(
        [
            [2, 2, 1],
            [2, 2, 1],
            [2, 1, 1],
            [1, 1, 2],
        ]
    )
    np.testing.assert_array_equal(
        conservative_residual_majority(main, expert, members), [2, 1, 1]
    )


def test_residual_expert_features_are_finite_and_aligned():
    main = np.asarray([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]], dtype=np.float32)
    expert = np.asarray([[0.4, 0.6], [0.2, 0.8], [0.7, 0.3]], dtype=np.float32)
    frame = pd.DataFrame(
        {
            "participant": ["p1"] * 3,
            "recording_id": ["r1"] * 3,
            "center": [10, 20, 30],
        }
    )
    features, names = build_residual_expert_features(
        main, expert, main.argmax(1), expert.argmax(1), frame
    )
    assert features.shape == (3, len(names))
    assert np.isfinite(features).all()


def test_disagreement_aware_features_add_member_evidence():
    main = np.asarray([[0.6, 0.4], [0.4, 0.6]], dtype=np.float32)
    expert = np.asarray([[0.3, 0.7], [0.8, 0.2]], dtype=np.float32)
    members = np.asarray(
        [
            [[0.9, 0.1], [0.3, 0.7]],
            [[0.3, 0.7], [0.5, 0.5]],
            [[0.6, 0.4], [0.4, 0.6]],
        ],
        dtype=np.float32,
    )
    labels = members.argmax(axis=2)
    frame = pd.DataFrame(
        {
            "participant": ["p1"] * 2,
            "recording_id": ["r1"] * 2,
            "center": [10, 20],
        }
    )
    plain, _ = build_residual_expert_features(
        main, expert, main.argmax(1), expert.argmax(1), frame
    )
    aware, names = build_residual_expert_features(
        main,
        expert,
        main.argmax(1),
        expert.argmax(1),
        frame,
        member_probabilities=members,
        member_predictions=labels,
    )
    assert aware.shape[0] == 2
    assert aware.shape[1] == len(names)
    assert aware.shape[1] > plain.shape[1]
    assert np.isfinite(aware).all()


def test_auxiliary_expert_evidence_adds_finite_features():
    main = np.asarray([[0.7, 0.3], [0.4, 0.6]], dtype=np.float32)
    expert = np.asarray([[0.2, 0.8], [0.7, 0.3]], dtype=np.float32)
    auxiliary = np.asarray([[0.3, 0.7], [0.6, 0.4]], dtype=np.float32)
    frame = pd.DataFrame(
        {
            "participant": ["p1"] * 2,
            "recording_id": ["r1"] * 2,
            "center": [10, 20],
        }
    )
    features, names = build_residual_expert_features(
        main,
        expert,
        main.argmax(1),
        expert.argmax(1),
        frame,
        auxiliary_probabilities=auxiliary,
        auxiliary_predictions=auxiliary.argmax(1),
    )
    assert features.shape == (2, len(names))
    assert np.isfinite(features).all()


def test_residual_expert_targets_and_application():
    main = np.asarray([0, 0, 1, 1])
    expert = np.asarray([1, 1, 0, 1])
    targets = np.asarray([0, 1, 0, 1])
    decisive, labels = residual_expert_training_targets(main, expert, targets)
    np.testing.assert_array_equal(decisive, [True, True, True, False])
    np.testing.assert_array_equal(labels, [1, 0, 0])
    routed = apply_residual_expert_gate(
        main, expert, np.asarray([0.8, 0.2, 0.1, 0.0]), 0.5
    )
    np.testing.assert_array_equal(routed, [0, 1, 0, 1])


def test_threshold_selection_falls_back_to_main():
    main = np.asarray([0, 0, 1, 1])
    expert = np.asarray([1, 1, 0, 0])
    targets = main.copy()
    threshold, scores = select_residual_expert_threshold(
        main,
        expert,
        np.asarray([0.1, 0.2, 0.3, 0.4]),
        targets,
        [0.25, 0.5, 0.75],
        2,
        minimum_gain=1e-3,
    )
    assert threshold == 0.0
    assert scores[-1]["expert_switch_fraction"] == 0.0


def test_utility_targets_and_rule_selection():
    main = np.asarray([0, 0, 1, 1, 0])
    expert = np.asarray([1, 1, 0, 1, 1])
    targets = np.asarray([0, 1, 2, 1, 2])
    selected, labels = residual_utility_training_targets(main, expert, targets)
    np.testing.assert_array_equal(selected, [True, True, True, False, True])
    np.testing.assert_array_equal(labels, [1, 2, 0, 0])
    probabilities = np.asarray(
        [
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    routed, utility = apply_residual_utility_gate(
        main, expert, probabilities, 1.0, 0.0
    )
    np.testing.assert_array_equal(routed, [0, 1, 1, 1, 1])
    assert utility.shape == main.shape
    loss_cost, threshold, scores = select_residual_utility_rule(
        main,
        expert,
        probabilities,
        targets,
        [0.5, 1.0],
        [0.0, 0.2],
        3,
    )
    assert loss_cost in {0.5, 1.0}
    assert threshold in {0.0, 0.2}
    assert scores


def test_pair_prior_offsets_favor_reliable_expert_direction():
    main = np.asarray([0, 0, 0, 0, 1, 1])
    expert = np.asarray([1, 1, 1, 1, 0, 0])
    targets = np.asarray([1, 1, 1, 0, 1, 1])
    offsets = fit_pair_expert_logit_offsets(main, expert, targets, 2, 2.0)
    raw = np.full(len(main), 0.5, dtype=np.float32)
    adjusted = apply_pair_expert_logit_offsets(
        raw, main, expert, offsets, 1.0, 2
    )
    assert adjusted[0] < 0.5
    assert adjusted[-1] > 0.5
    assert np.isfinite(adjusted).all()
