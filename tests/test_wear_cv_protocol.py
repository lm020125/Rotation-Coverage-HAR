from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_grouped_cv import aggregate_fold_metrics, load_completed_fold


def test_final_wear_folds_match_frozen_main_experiment_protocol() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "wear_paper_folds.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["test_folds"] == [
        ["sbj_0", "sbj_12", "sbj_14", "sbj_15", "sbj_3", "sbj_5"],
        ["sbj_1", "sbj_10", "sbj_11", "sbj_16", "sbj_4", "sbj_6"],
        ["sbj_13", "sbj_17", "sbj_2", "sbj_7", "sbj_8", "sbj_9"],
    ]
    flattened = [participant for fold in payload["test_folds"] for participant in fold]
    assert len(flattened) == len(set(flattened)) == 18
    assert set(flattened) == {f"sbj_{index}" for index in range(18)}


def test_fold_aggregation_pools_center_and_samplewise_confusions() -> None:
    folds = [
        {
            "confusion_matrix": [[3, 1], [0, 2]],
            "samplewise": {
                "confusion_matrix": [[30, 10], [0, 20]],
                "per_participant": {"P1": {"macro_f1": 0.8}},
            },
        },
        {
            "confusion_matrix": [[1, 0], [1, 3]],
            "samplewise": {
                "confusion_matrix": [[10, 0], [10, 30]],
                "per_participant": {"P2": {"macro_f1": 0.7}},
            },
        },
    ]
    aggregate = aggregate_fold_metrics(folds)
    assert aggregate["confusion_matrix"] == [[4, 1], [1, 5]]
    assert aggregate["samplewise"]["confusion_matrix"] == [[40, 10], [10, 50]]
    assert set(aggregate["samplewise"]["per_participant"]) == {"P1", "P2"}


def test_resume_completed_fold_requires_matching_saved_protocol(tmp_path: Path) -> None:
    fold_output = tmp_path / "fold_1"
    fold_output.mkdir()
    split = {
        "test_participants": ["p1"],
        "val_participants_for_epoch_selection": ["p2"],
    }
    metrics = {
        "macro_f1": 0.8,
        "test_participants": ["p1"],
        "val_participants": ["p2"],
    }
    (fold_output / "split.json").write_text(json.dumps(split), encoding="utf-8")
    (fold_output / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    assert load_completed_fold(fold_output, ["p1"], ["p2"]) == metrics

    with pytest.raises(ValueError, match="does not match requested"):
        load_completed_fold(fold_output, ["p3"], ["p2"])
