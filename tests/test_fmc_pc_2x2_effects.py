from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_fmc_pc_2x2_effects import (  # noqa: E402
    analyze_summary,
    factorial_effects,
    paired_bootstrap_ci,
    paired_cell_values,
)


def synthetic_summary() -> dict:
    values = {
        "uniform_no_pc": [0.10, 0.20, 0.30],
        "uniform_pc": [0.20, 0.30, 0.40],
        "empirical_no_pc": [0.40, 0.50, 0.60],
        "empirical_pc": [0.80, 0.90, 1.00],
    }
    return {
        "protocol": {"seeds": [256, 3407, 114514]},
        "methods": {
            method: {
                "scenarios": {
                    "valid_signal_self": {
                        "macro_f1": {"values": row},
                        "accuracy": {"values": row},
                    }
                }
            }
            for method, row in values.items()
        },
    }


def test_factorial_effects_are_paired_within_seed() -> None:
    cells = np.asarray(
        [
            [0.10, 0.20, 0.40, 0.80],
            [0.20, 0.30, 0.50, 0.90],
            [0.30, 0.40, 0.60, 1.00],
        ]
    )
    effects = factorial_effects(cells)
    np.testing.assert_allclose(
        effects["empirical_amplitude_main_effect"],
        [0.45, 0.45, 0.45],
    )
    np.testing.assert_allclose(
        effects["prediction_consistency_main_effect"],
        [0.25, 0.25, 0.25],
    )
    np.testing.assert_allclose(
        effects["amplitude_x_pc_interaction"],
        [0.30, 0.30, 0.30],
    )


def test_bootstrap_is_reproducible_with_fixed_seed() -> None:
    values = np.asarray([0.1, 0.2, 0.4])
    first = paired_bootstrap_ci(values, repeats=2_000, random_seed=20260723)
    second = paired_bootstrap_ci(values, repeats=2_000, random_seed=20260723)
    assert first == second


def test_analysis_marks_three_seed_intervals_as_descriptive_only() -> None:
    result = analyze_summary(
        synthetic_summary(),
        scenario="valid_signal_self",
        metrics=("macro_f1", "accuracy"),
        bootstrap_repeats=1_000,
        bootstrap_seed=20260723,
    )
    assert result["n_seeds"] == 3
    assert result["interpretation_guardrail"]["status"] == "descriptive_only"
    assert not result["interpretation_guardrail"]["p_values_computed"]
    assert not result["interpretation_guardrail"]["significance_claim_authorized"]
    effect = result["metrics"]["macro_f1"]["effects"][
        "amplitude_x_pc_interaction"
    ]
    assert np.isclose(effect["mean_percentage_points"], 30.0)


def test_cell_value_validation_rejects_missing_seed_value() -> None:
    summary = synthetic_summary()
    summary["methods"]["empirical_pc"]["scenarios"]["valid_signal_self"][
        "macro_f1"
    ]["values"] = [0.8, 0.9]
    try:
        paired_cell_values(summary, "valid_signal_self", "macro_f1")
    except ValueError as error:
        assert "expected one value for each of 3 seeds" in str(error)
    else:
        raise AssertionError("Missing seed value was accepted")
