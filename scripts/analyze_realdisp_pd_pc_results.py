from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcbd_har.metrics import classification_metrics_from_confusion  # noqa: E402


DEFAULT_METHODS = (
    "clean_rcbd",
    "random_rotation",
    "pd_rotation_only",
    "pd_rotation_gain",
    "pd_full_no_pc",
    "random_rotation_pc",
    "pd_pc",
)
DEFAULT_SEEDS = (256, 3407, 114514)
SCENARIOS = ("final_ideal_subjects13_17", "final_self_subjects13_17")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


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


def paired_effect(
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    participants: list[str],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    differences = np.asarray(
        [candidate[item]["mean"] - baseline[item]["mean"] for item in participants],
        dtype=np.float64,
    )
    sample_std = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    return {
        "unit": "participant after averaging seeds",
        "participants": participants,
        "differences": differences.tolist(),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "bootstrap_95_ci": paired_bootstrap_interval(
            differences, samples=bootstrap_samples, seed=bootstrap_seed
        ),
        "exact_two_sided_sign_flip_pvalue": exact_sign_flip_pvalue(differences),
        "paired_cohens_dz": (
            float(differences.mean() / sample_std) if sample_std > 0.0 else None
        ),
        "improved_participants": int(np.sum(differences > 0.0)),
        "harmed_participants": int(np.sum(differences < 0.0)),
        "unchanged_participants": int(np.sum(differences == 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Participant-safe statistics for frozen REALDISP PD-PC results."
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--exclude-damaged", nargs="+", default=["subject13"])
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    methods = [str(value) for value in args.methods]
    seeds = [int(value) for value in args.seeds]
    payload: dict[str, Any] = {
        "protocol": {
            "independent_unit": "participant",
            "seed_handling": "average within participant before inference",
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "damaged_records_excluded_only_in_sensitivity": args.exclude_damaged,
            "methods": methods,
            "seeds": seeds,
        },
        "scenarios": {},
    }
    participant_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []

    for scenario_index, scenario in enumerate(SCENARIOS):
        loaded: dict[str, dict[int, dict[str, Any]]] = {}
        for method in methods:
            loaded[method] = {}
            for seed in seeds:
                path = input_root / method / f"seed{seed}" / scenario / "metrics.json"
                if not path.exists():
                    raise FileNotFoundError(path)
                loaded[method][seed] = read_json(path)

        reference = loaded[methods[0]][seeds[0]]
        participants = [str(value) for value in reference["participants"]]
        num_classes = len(reference["confusion_matrix"])
        scenario_payload: dict[str, Any] = {"methods": {}, "comparisons": {}}
        confusion_archive: dict[str, np.ndarray] = {}
        participant_by_method: dict[str, dict[str, dict[str, float]]] = {}

        for method in methods:
            participant_by_method[method] = {}
            for participant in participants:
                values = [
                    float(loaded[method][seed]["per_participant"][participant]["macro_f1"])
                    for seed in seeds
                ]
                participant_by_method[method][participant] = summarize(values)
                participant_rows.append(
                    {
                        "scenario": scenario,
                        "participant": participant,
                        "method": method,
                        **{f"seed_{seed}": values[index] for index, seed in enumerate(seeds)},
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=0)),
                    }
                )

            per_class: dict[str, Any] = {}
            for class_index in range(num_classes):
                values = [
                    float(
                        loaded[method][seed]["per_class"][str(class_index)]["f1-score"]
                    )
                    for seed in seeds
                ]
                per_class[str(class_index)] = summarize(values)
                class_rows.append(
                    {
                        "scenario": scenario,
                        "class_index": class_index,
                        "activity": f"A{class_index + 1}",
                        "method": method,
                        "mean_f1": float(np.mean(values)),
                        "std_f1": float(np.std(values, ddof=0)),
                    }
                )

            sensitivity_values = []
            normalized_confusions = []
            for seed in seeds:
                remaining = [
                    participant
                    for participant in participants
                    if participant not in set(args.exclude_damaged)
                ]
                sensitivity_matrix = np.sum(
                    [
                        np.asarray(
                            loaded[method][seed]["per_participant"][participant][
                                "confusion_matrix"
                            ],
                            dtype=np.int64,
                        )
                        for participant in remaining
                    ],
                    axis=0,
                )
                sensitivity_values.append(
                    float(
                        classification_metrics_from_confusion(sensitivity_matrix)[
                            "macro_f1"
                        ]
                    )
                )
                confusion = np.asarray(
                    loaded[method][seed]["confusion_matrix"], dtype=np.float64
                )
                row_sum = confusion.sum(axis=1, keepdims=True)
                normalized_confusions.append(
                    np.divide(
                        confusion,
                        row_sum,
                        out=np.zeros_like(confusion),
                        where=row_sum > 0,
                    )
                )
            confusion_archive[method] = np.mean(normalized_confusions, axis=0)
            scenario_payload["methods"][method] = {
                "pooled_macro_f1": summarize(
                    [float(loaded[method][seed]["macro_f1"]) for seed in seeds]
                ),
                "pooled_accuracy": summarize(
                    [float(loaded[method][seed]["accuracy"]) for seed in seeds]
                ),
                "participant_macro_f1": participant_by_method[method],
                "per_class_f1": per_class,
                "sensitivity_excluding_damaged_macro_f1": summarize(
                    sensitivity_values
                ),
            }

        comparisons = [
            ("clean_rcbd", "random_rotation"),
            ("clean_rcbd", "pd_pc"),
            ("random_rotation", "pd_pc"),
        ]
        for comparison_index, (baseline, candidate) in enumerate(comparisons):
            if baseline not in methods or candidate not in methods:
                continue
            key = f"{candidate}_vs_{baseline}"
            inference = paired_effect(
                participant_by_method[baseline],
                participant_by_method[candidate],
                participants,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=(
                    args.bootstrap_seed + scenario_index * 100 + comparison_index
                ),
            )
            pooled_differences = [
                float(loaded[candidate][seed]["macro_f1"])
                - float(loaded[baseline][seed]["macro_f1"])
                for seed in seeds
            ]
            inference["descriptive_pooled_macro_f1_difference"] = summarize(
                pooled_differences
            )
            scenario_payload["comparisons"][key] = inference

        np.savez_compressed(output / f"{scenario}_row_normalized_confusions.npz", **confusion_archive)
        payload["scenarios"][scenario] = scenario_payload

    write_json(payload, output / "statistical_analysis.json")
    for filename, rows in (
        ("participant_metrics.csv", participant_rows),
        ("class_metrics.csv", class_rows),
    ):
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    self_results = payload["scenarios"]["final_self_subjects13_17"]
    lines = [
        "# REALDISP PD-PC statistical audit",
        "",
        "Independent unit: participant; seeds are averaged within participant.",
        "",
        "| Comparison | Pooled Macro-F1 delta | Participant mean delta | 95% bootstrap CI | Exact sign-flip p |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, comparison in self_results["comparisons"].items():
        pooled = comparison["descriptive_pooled_macro_f1_difference"]["mean"]
        low, high = comparison["bootstrap_95_ci"]
        lines.append(
            f"| {name} | {pooled:.4%} | {comparison['mean_difference']:.4%} | "
            f"[{low:.4%}, {high:.4%}] | "
            f"{comparison['exact_two_sided_sign_flip_pvalue']:.4f} |"
        )
    (output / "paper_statistics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote participant-safe audit to {output}", flush=True)


if __name__ == "__main__":
    main()
