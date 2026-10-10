from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_realdisp_pd_pc_final_evaluation import summarize_final  # noqa: E402


def test_final_summary_keeps_seed_and_participant_units_distinct() -> None:
    methods = {}
    for method, offset in (("pd_rcbd", 0.0), ("pd_prediction_consistency", 0.05)):
        methods[method] = {}
        for seed in ("256", "3407", "114514"):
            methods[method][seed] = {
                scenario: {
                    "macro_f1": 0.70 + offset,
                    "accuracy": 0.72 + offset,
                    "per_participant_macro_f1": {
                        "subject13": 0.60 + offset,
                        "subject14": 0.80 + offset,
                    },
                }
                for scenario in ("ideal", "self")
            }

    summary = summarize_final(methods)

    assert summary["self"]["pd_prediction_consistency"]["macro_f1"]["mean"] == 0.75
    assert abs(summary["self"]["pd_pc_minus_pd_rcbd_pp"]["macro_f1_mean"] - 5.0) < 1e-9
    assert abs(
        summary["self"]["participant_mean_macro_f1"]["subject13"][
            "pd_pc_minus_pd_rcbd_pp"
        ]
        - 5.0
    ) < 1e-9
