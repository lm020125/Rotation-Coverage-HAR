import numpy as np
import pandas as pd

from rcbd_har.multi_expert_selector import (
    apply_multi_expert_selector,
    build_multi_expert_features,
    multi_expert_targets,
)


def test_multi_expert_targets_use_unique_corrections():
    main = np.asarray([0, 0, 0, 1])
    raw = np.asarray([1, 1, 2, 1])
    relational = np.asarray([2, 2, 1, 0])
    targets = np.asarray([0, 1, 1, 1])
    np.testing.assert_array_equal(
        multi_expert_targets(main, raw, relational, targets), [0, 1, 2, 0]
    )


def test_multi_expert_features_and_application():
    frame = pd.DataFrame(
        {
            "participant": ["p1"] * 3,
            "recording_id": ["r1"] * 3,
            "center": [10, 20, 30],
        }
    )
    main = np.asarray([0, 0, 1])
    raw = np.asarray([1, 0, 1])
    relational = np.asarray([0, 1, 0])
    features = build_multi_expert_features(
        main,
        raw,
        relational,
        np.asarray([0.2, 0.8, 0.7]),
        np.asarray([0.7, 0.2, 0.6]),
        frame,
        2,
    )
    assert features.shape[0] == 3
    assert np.isfinite(features).all()
    probabilities = np.asarray([[0.2, 0.7, 0.1], [0.6, 0.1, 0.3], [0.3, 0.2, 0.5]])
    selected = apply_multi_expert_selector(
        main, raw, relational, probabilities, margin=0.1
    )
    np.testing.assert_array_equal(selected, [1, 0, 0])
