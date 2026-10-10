from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_realdisp_pd_pc_seed_completion import summarize_multiseed  # noqa: E402


def test_pd_pc_frozen_gate_requires_stable_self_gain() -> None:
    methods = {
        "pd_rcbd": {
            seed: {
                "clean_macro_f1": 0.96,
                "rotation_stress_macro_f1": 0.88,
                "self_macro_f1": value,
            }
            for seed, value in {"256": 0.65, "3407": 0.70, "114514": 0.68}.items()
        },
        "pd_prediction_consistency": {
            seed: {
                "clean_macro_f1": 0.955,
                "rotation_stress_macro_f1": 0.90,
                "self_macro_f1": value,
            }
            for seed, value in {"256": 0.69, "3407": 0.76, "114514": 0.71}.items()
        },
    }

    aggregate, gate = summarize_multiseed(methods)

    assert gate["passes"] is True
    assert gate["positive_self_gain_seeds"] == 3
    assert aggregate["pd_prediction_consistency"]["self_macro_f1"]["mean"] > 0.7


def test_pd_pc_frozen_gate_rejects_single_seed_collapse() -> None:
    methods = {
        "pd_rcbd": {
            seed: {
                "clean_macro_f1": 0.96,
                "rotation_stress_macro_f1": 0.88,
                "self_macro_f1": 0.60,
            }
            for seed in ("256", "3407", "114514")
        },
        "pd_prediction_consistency": {
            seed: {
                "clean_macro_f1": 0.96,
                "rotation_stress_macro_f1": 0.90,
                "self_macro_f1": value,
            }
            for seed, value in {"256": 0.70, "3407": 0.70, "114514": 0.55}.items()
        },
    }

    _, gate = summarize_multiseed(methods)

    assert gate["mean_self_gain_pp"] >= 3.0
    assert gate["passes"] is False
    assert gate["minimum_self_gain_pp"] < -2.0
