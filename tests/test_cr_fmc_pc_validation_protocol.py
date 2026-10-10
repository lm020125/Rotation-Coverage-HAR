from scripts.run_realdisp_cr_fmc_pc_validation import per_class_f1


def test_per_class_f1_ignores_aggregate_report_rows() -> None:
    payload = {
        "per_class": {
            "0": {"f1-score": 0.25},
            "1": {"f1-score": 0.75},
            "accuracy": 0.5,
            "macro avg": {"f1-score": 0.5},
            "weighted avg": {"f1-score": 0.6},
        }
    }

    assert per_class_f1(payload).tolist() == [0.25, 0.75]
