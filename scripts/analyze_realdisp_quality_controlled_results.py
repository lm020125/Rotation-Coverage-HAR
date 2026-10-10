"""Build the quality-controlled REALDISP analysis without retraining models.

The analysis uses only frozen predictions and a label-free raw-input eligibility
rule. A Self-placement participant-session is invalid when every one of the four
distal-limb acceleration streams has more than a selected fraction of near-zero
vector norms (<= 1e-6). Subject 13 is the only final-test session meeting the
rule at thresholds 0.50, 0.80, and 0.95.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHOD_LABELS = {
    "clean_rcbd": "Clean RCBD",
    "random_rotation": "Random rotation",
    "pd_rotation_only": "PD rotation",
    "pd_rotation_gain": "PD rotation + gain",
    "pd_full_no_pc": "PD full (no PC)",
    "random_rotation_pc": "Random rotation + PC",
    "pd_pc": "PD-PC",
}
MAIN_METHODS = ("clean_rcbd", "random_rotation", "pd_pc")
VALID_PARTICIPANTS = ("subject14", "subject15", "subject16", "subject17")
SEEDS = (256, 3407, 114514)
THRESHOLDS = (0.50, 0.80, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistical-json", type=Path, required=True)
    parser.add_argument("--participant-csv", type=Path, required=True)
    parser.add_argument("--dcl-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize(values: list[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
    }


def paired_bootstrap_interval(
    differences: np.ndarray, *, samples: int, seed: int
) -> list[float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, len(differences), size=(samples, len(differences))
    )
    means = differences[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    observed = abs(float(differences.mean()))
    statistics = [
        abs(float(np.mean(differences * np.asarray(signs, dtype=np.float64))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, float]:
    confusion = np.asarray(confusion, dtype=np.float64)
    true_positive = np.diag(confusion)
    actual = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        actual,
        out=np.zeros_like(true_positive),
        where=actual > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    total = confusion.sum()
    return {
        "accuracy": float(true_positive.sum() / total) if total > 0 else 0.0,
        "macro_f1": float(f1.mean()),
    }


def main() -> None:
    args = parse_args()
    payload = read_json(args.statistical_json.resolve())
    participant = pd.read_csv(args.participant_csv.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for scenario in ("final_ideal_subjects13_17", "final_self_subjects13_17"):
        scenario_payload = payload["scenarios"][scenario]["methods"]
        for order, (method, method_payload) in enumerate(scenario_payload.items()):
            valid = method_payload["sensitivity_excluding_damaged_macro_f1"]
            released = method_payload["pooled_macro_f1"]
            summary_rows.append(
                {
                    "scenario": scenario.replace("_subjects13_17", ""),
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "method_order": order,
                    "valid_signal_participants": "subjects14-17",
                    "valid_signal_n": 4,
                    "valid_signal_macro_f1_mean": valid["mean"] * 100,
                    "valid_signal_macro_f1_std": valid["std"] * 100,
                    "as_released_participants": "subjects13-17",
                    "as_released_n": 5,
                    "as_released_macro_f1_mean": released["mean"] * 100,
                    "as_released_macro_f1_std": released["std"] * 100,
                    "as_released_accuracy_mean": method_payload["pooled_accuracy"]["mean"]
                    * 100,
                    "as_released_accuracy_std": method_payload["pooled_accuracy"]["std"]
                    * 100,
                }
            )
            for seed, valid_value, released_value in zip(
                SEEDS, valid["values"], released["values"], strict=True
            ):
                run_rows.append(
                    {
                        "scenario": scenario.replace("_subjects13_17", ""),
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "method_order": order,
                        "seed": seed,
                        "valid_signal_macro_f1": valid_value * 100,
                        "as_released_macro_f1": released_value * 100,
                    }
                )

    summary = pd.DataFrame(summary_rows)
    runs = pd.DataFrame(run_rows)
    summary.to_csv(output / "quality_controlled_summary.csv", index=False, float_format="%.8f")
    runs.to_csv(output / "quality_controlled_runs.csv", index=False, float_format="%.8f")

    self_rows = participant.loc[
        (participant["scenario"] == "final_self_subjects13_17")
        & participant["participant"].isin(VALID_PARTICIPANTS)
        & participant["method"].isin(MAIN_METHODS)
    ].copy()
    comparisons = (
        ("clean_rcbd", "random_rotation"),
        ("clean_rcbd", "pd_pc"),
        ("random_rotation", "pd_pc"),
    )
    comparison_rows: list[dict[str, Any]] = []
    for index, (baseline, candidate) in enumerate(comparisons):
        baseline_values = (
            self_rows.loc[self_rows["method"] == baseline]
            .set_index("participant")["mean"]
            .reindex(VALID_PARTICIPANTS)
        )
        candidate_values = (
            self_rows.loc[self_rows["method"] == candidate]
            .set_index("participant")["mean"]
            .reindex(VALID_PARTICIPANTS)
        )
        differences = (candidate_values - baseline_values).to_numpy(dtype=np.float64)
        sample_std = float(differences.std(ddof=1))
        low, high = paired_bootstrap_interval(
            differences,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + index,
        )
        baseline_summary = summary.loc[
            (summary["scenario"] == "final_self") & (summary["method"] == baseline)
        ].iloc[0]
        candidate_summary = summary.loc[
            (summary["scenario"] == "final_self") & (summary["method"] == candidate)
        ].iloc[0]
        comparison_rows.append(
            {
                "comparison": f"{METHOD_LABELS[candidate]} vs {METHOD_LABELS[baseline]}",
                "participants": "subjects14-17",
                "n_participants": len(VALID_PARTICIPANTS),
                "pooled_macro_f1_delta_pp": candidate_summary[
                    "valid_signal_macro_f1_mean"
                ]
                - baseline_summary["valid_signal_macro_f1_mean"],
                "participant_equal_mean_delta_pp": float(differences.mean() * 100),
                "bootstrap_95_ci_low_pp": low * 100,
                "bootstrap_95_ci_high_pp": high * 100,
                "exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(differences),
                "paired_cohens_dz": (
                    float(differences.mean() / sample_std) if sample_std > 0 else np.nan
                ),
                "improved": int(np.sum(differences > 0)),
                "harmed": int(np.sum(differences < 0)),
                "unchanged": int(np.sum(differences == 0)),
            }
        )
    comparisons_frame = pd.DataFrame(comparison_rows)
    comparisons_frame.to_csv(
        output / "quality_controlled_participant_comparisons.csv",
        index=False,
        float_format="%.8f",
    )

    if args.dcl_root is not None:
        dcl_rows: list[dict[str, Any]] = []
        dcl_root = args.dcl_root.resolve()
        for method in ("dcl_clean", "dcl_random_rotation", "dcl_pd_pc"):
            for scenario in ("final_ideal_subjects13_17", "final_self_subjects13_17"):
                valid_macro_f1: list[float] = []
                valid_accuracy: list[float] = []
                released_macro_f1: list[float] = []
                released_accuracy: list[float] = []
                for seed in SEEDS:
                    metrics = read_json(
                        dcl_root / method / f"seed{seed}" / scenario / "metrics.json"
                    )
                    valid_confusion = np.sum(
                        [
                            np.asarray(
                                metrics["per_participant"][participant][
                                    "confusion_matrix"
                                ],
                                dtype=np.int64,
                            )
                            for participant in VALID_PARTICIPANTS
                        ],
                        axis=0,
                    )
                    valid_metrics = metrics_from_confusion(valid_confusion)
                    valid_macro_f1.append(valid_metrics["macro_f1"] * 100)
                    valid_accuracy.append(valid_metrics["accuracy"] * 100)
                    released_macro_f1.append(float(metrics["macro_f1"]) * 100)
                    released_accuracy.append(float(metrics["accuracy"]) * 100)
                dcl_rows.append(
                    {
                        "scenario": scenario.replace("_subjects13_17", ""),
                        "method": method,
                        "valid_signal_macro_f1_mean": np.mean(valid_macro_f1),
                        "valid_signal_macro_f1_std": np.std(valid_macro_f1, ddof=0),
                        "valid_signal_accuracy_mean": np.mean(valid_accuracy),
                        "valid_signal_accuracy_std": np.std(valid_accuracy, ddof=0),
                        "as_released_macro_f1_mean": np.mean(released_macro_f1),
                        "as_released_macro_f1_std": np.std(released_macro_f1, ddof=0),
                        "as_released_accuracy_mean": np.mean(released_accuracy),
                        "as_released_accuracy_std": np.std(released_accuracy, ddof=0),
                    }
                )
        pd.DataFrame(dcl_rows).to_csv(
            output / "quality_controlled_deepconvlstm_summary.csv",
            index=False,
            float_format="%.8f",
        )

    pd_pc_self = summary.loc[
        (summary["scenario"] == "final_self") & (summary["method"] == "pd_pc")
    ].iloc[0]
    threshold_rows = []
    for threshold in THRESHOLDS:
        threshold_rows.append(
            {
                "near_zero_norm_epsilon": 1e-6,
                "per_stream_zero_fraction_threshold": threshold,
                "participant_session_rule": "all four target streams exceed threshold",
                "excluded_participants": "subject13",
                "included_participants": "subjects14-17",
                "included_n": 4,
                "pd_pc_self_macro_f1_mean": pd_pc_self[
                    "valid_signal_macro_f1_mean"
                ],
                "pd_pc_self_macro_f1_std": pd_pc_self[
                    "valid_signal_macro_f1_std"
                ],
            }
        )
    pd.DataFrame(threshold_rows).to_csv(
        output / "quality_threshold_sensitivity.csv",
        index=False,
        float_format="%.8f",
    )

    audit = {
        "analysis_type": "post hoc, label-free raw-input quality sensitivity",
        "retraining_performed": False,
        "model_or_prediction_changed": False,
        "eligibility_rule": {
            "near_zero_vector_norm_epsilon": 1e-6,
            "stream_fails_if_zero_fraction_strictly_greater_than": 0.50,
            "participant_session_fails_if": "all four target streams fail",
        },
        "raw_input_evidence": {
            "subject13_self": "all four target distal-limb streams are 100% zero",
            "subjects14_17_self": "pass the 0.50 rule",
        },
        "thresholds_checked": list(THRESHOLDS),
        "same_exclusion_at_all_thresholds": ["subject13"],
        "post_hoc_valid_signal_conditional_result": {
            "participants": list(VALID_PARTICIPANTS),
            "metric": "sample-pooled Macro-F1",
            "mean_percent": float(pd_pc_self["valid_signal_macro_f1_mean"]),
            "std_percent": float(pd_pc_self["valid_signal_macro_f1_std"]),
        },
        "frozen_as_released_benchmark": {
            "participants": [f"subject{i}" for i in range(13, 18)],
            "metric": "sample-pooled Macro-F1",
            "mean_percent": float(pd_pc_self["as_released_macro_f1_mean"]),
            "std_percent": float(pd_pc_self["as_released_macro_f1_std"]),
        },
        "important_boundary": (
            "The eligibility rule was formalized after the frozen evaluation. "
            "The post hoc valid-signal result is conditional and must be paired "
            "with the complete frozen as-released benchmark."
        ),
    }
    with (output / "quality_controlled_analysis.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
