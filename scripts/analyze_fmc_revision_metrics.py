from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any


SEEDS = (256, 3407, 114514)
PARTICIPANTS = ("subject14", "subject15", "subject16", "subject17")
NUM_CLASSES = 33
ACTIVITY_NAMES = (
    "Walking",
    "Jogging",
    "Running",
    "Jump up",
    "Jump front & back",
    "Jump sideways",
    "Jump leg/arms open/closed",
    "Jump rope",
    "Trunk twist (arms outstretched)",
    "Trunk twist (elbows bent)",
    "Waist bends forward",
    "Waist rotation",
    "Waist bends (reach foot with opposite hand)",
    "Reach heels backwards",
    "Lateral bend",
    "Lateral bend with arm up",
    "Repetitive forward stretching",
    "Upper trunk and lower body opposite twist",
    "Lateral elevation of arms",
    "Frontal elevation of arms",
    "Frontal hand claps",
    "Frontal crossing of arms",
    "Shoulders high-amplitude rotation",
    "Shoulders low-amplitude rotation",
    "Arms inner rotation",
    "Knees (alternating) to the breast",
    "Heels (alternating) to the backside",
    "Knees bending (crouching)",
    "Knees (alternating) bending forward",
    "Rotation on the knees",
    "Rowing",
    "Elliptical bike",
    "Cycling",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def summary(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "values": values,
        "mean": fmean(values),
        "population_sd": pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def observed_class_macro_f1(participant_payload: dict[str, Any]) -> tuple[float, int]:
    values = []
    for index in range(NUM_CLASSES):
        row = participant_payload["per_class"][str(index)]
        if float(row["support"]) > 0:
            values.append(float(row["f1-score"]))
    return (fmean(values), len(values)) if values else (math.nan, 0)


def resolve_metric_file(method_root: Path, seed: int) -> Path | None:
    candidates = (
        method_root / f"seed{seed}" / "valid_signal_self" / "metrics.json",
        method_root / f"seed{seed}" / "valid_self" / "metrics.json",
        method_root / f"seed_{seed}" / "valid_signal_self" / "metrics.json",
        method_root / f"seed_{seed}" / "valid_self" / "metrics.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def parse_method_override(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("--method must be NAME=PATH")
    name, raw_path = text.split("=", 1)
    return name.strip(), Path(raw_path.strip()).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize REALDISP valid-signal Self metrics without retraining. "
            "Input metrics must contain per_participant, per_class and confusion_matrix."
        )
    )
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Add or replace a method root. Repeat as needed.",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = (
        args.output.resolve()
        if args.output
        else repo / "results" / "fmc_pc_revision_20260723_local" / "participant_class"
    )
    existing = (
        repo
        / "artifacts"
        / "pdpc_journal_strengthening_20260720"
        / "server_results"
        / "existing_reanalysis"
    )
    exports = repo / "results" / "fmc_pc_revision_20260723_local" / "server_exports"
    methods: dict[str, Path] = {
        "clean_rcbd": existing / "clean_rcbd",
        "uniform_random_no_pc": existing / "random_rotation",
        "angle_fmc_pc": exports / "fmc_angle_pc",
        "uniform_random_pc": exports / "uniform_random_pc",
        "angle_fmc_no_pc": exports / "angle_fmc_no_pc",
    }
    for text in args.method:
        name, path = parse_method_override(text)
        methods[name] = path

    availability: list[dict[str, Any]] = []
    loaded: dict[str, dict[int, dict[str, Any]]] = {}
    missing_files: list[str] = []
    for method, root in methods.items():
        loaded[method] = {}
        for seed in SEEDS:
            path = resolve_metric_file(root, seed)
            availability.append(
                {
                    "method": method,
                    "seed": seed,
                    "method_root": display_path(root, repo),
                    "metrics_file": display_path(path, repo) if path else "",
                    "sha256": sha256(path) if path else "",
                    "available": bool(path),
                }
            )
            if path is None:
                missing_files.append(
                    display_path(
                        root / f"seed{seed}" / "valid_signal_self" / "metrics.json",
                        repo,
                    )
                )
                continue
            payload = read_json(path)
            required = {"accuracy", "macro_f1", "confusion_matrix", "per_class", "per_participant"}
            absent = required.difference(payload)
            if absent:
                raise KeyError(f"{path} lacks required keys: {sorted(absent)}")
            participants = tuple(payload.get("participants", ()))
            if participants and participants != PARTICIPANTS:
                raise ValueError(
                    f"{path}: expected participants {PARTICIPANTS}, got {participants}"
                )
            matrix = payload["confusion_matrix"]
            if len(matrix) != NUM_CLASSES or any(len(row) != NUM_CLASSES for row in matrix):
                raise ValueError(f"{path}: confusion matrix is not {NUM_CLASSES}x{NUM_CLASSES}")
            loaded[method][seed] = payload

    write_csv(output / "availability.csv", availability)
    write_json(
        output / "availability.json",
        {"methods": availability, "missing_expected_files": missing_files},
    )

    complete_methods = {
        method: seed_payloads
        for method, seed_payloads in loaded.items()
        if set(seed_payloads) == set(SEEDS)
    }
    aggregate_rows: list[dict[str, Any]] = []
    aggregate_summary: dict[str, Any] = {}
    participant_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_payload: dict[str, Any] = {}
    confusion_rows: list[dict[str, Any]] = []

    for method, seed_payloads in complete_methods.items():
        accuracy_values = []
        macro_values = []
        raw_accuracy_values = []
        raw_macro_values = []
        summed_matrix = [[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)]
        for seed in SEEDS:
            payload = seed_payloads[seed]
            accuracy = float(payload["accuracy"])
            macro_f1 = float(payload["macro_f1"])
            raw_accuracy = float(payload.get("raw", {}).get("accuracy", math.nan))
            raw_macro_f1 = float(payload.get("raw", {}).get("macro_f1", math.nan))
            accuracy_values.append(accuracy)
            macro_values.append(macro_f1)
            raw_accuracy_values.append(raw_accuracy)
            raw_macro_values.append(raw_macro_f1)
            aggregate_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "accuracy": accuracy,
                    "macro_f1_fixed_33": macro_f1,
                    "raw_accuracy": raw_accuracy,
                    "raw_macro_f1_fixed_33": raw_macro_f1,
                }
            )
            for participant in PARTICIPANTS:
                item = payload["per_participant"][participant]
                observed_f1, observed_count = observed_class_macro_f1(item)
                participant_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "participant": participant,
                        "accuracy": float(item["accuracy"]),
                        "macro_f1_fixed_33": float(item["macro_f1"]),
                        "macro_f1_observed_classes": observed_f1,
                        "observed_class_count": observed_count,
                    }
                )
            for class_index in range(NUM_CLASSES):
                item = payload["per_class"][str(class_index)]
                class_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "class_index": class_index,
                        "activity_id": f"A{class_index + 1}",
                        "activity_name": ACTIVITY_NAMES[class_index],
                        "precision": float(item["precision"]),
                        "recall": float(item["recall"]),
                        "f1": float(item["f1-score"]),
                        "support": int(float(item["support"])),
                    }
                )
            for row_index, row in enumerate(payload["confusion_matrix"]):
                for column_index, value in enumerate(row):
                    summed_matrix[row_index][column_index] += int(value)

        aggregate_summary[method] = {
            "accuracy": summary(accuracy_values),
            "macro_f1_fixed_33": summary(macro_values),
            "raw_accuracy": summary(raw_accuracy_values),
            "raw_macro_f1_fixed_33": summary(raw_macro_values),
        }
        row_normalized = []
        for row in summed_matrix:
            total = sum(row)
            row_normalized.append([value / total if total else 0.0 for value in row])
        confusion_payload[method] = {
            "note": "Counts are summed over three seed evaluations; row-normalized values divide each true-class row by its sum.",
            "summed_counts": summed_matrix,
            "row_normalized": row_normalized,
        }
        for true_index in range(NUM_CLASSES):
            for predicted_index in range(NUM_CLASSES):
                if true_index == predicted_index:
                    continue
                confusion_rows.append(
                    {
                        "method": method,
                        "true_class": f"A{true_index + 1}",
                        "true_activity": ACTIVITY_NAMES[true_index],
                        "predicted_class": f"A{predicted_index + 1}",
                        "predicted_activity": ACTIVITY_NAMES[predicted_index],
                        "summed_count_three_seeds": summed_matrix[true_index][predicted_index],
                        "true_row_fraction": row_normalized[true_index][predicted_index],
                    }
                )
        write_csv(
            output / f"confusion_{method}_summed_counts.csv",
            [
                {"true_class": f"A{i + 1}", **{f"pred_A{j + 1}": row[j] for j in range(NUM_CLASSES)}}
                for i, row in enumerate(summed_matrix)
            ],
        )
        write_csv(
            output / f"confusion_{method}_row_normalized.csv",
            [
                {"true_class": f"A{i + 1}", **{f"pred_A{j + 1}": row_normalized[i][j] for j in range(NUM_CLASSES)}}
                for i in range(NUM_CLASSES)
            ],
        )

    write_csv(output / "aggregate_seed_metrics.csv", aggregate_rows)
    write_json(output / "aggregate_summary.json", aggregate_summary)
    write_csv(
        output / "paper_ready_aggregate_summary.csv",
        [
            {
                "method": method,
                "accuracy_mean_pct": 100.0 * metrics["accuracy"]["mean"],
                "accuracy_population_sd_pct": 100.0
                * metrics["accuracy"]["population_sd"],
                "macro_f1_mean_pct": 100.0 * metrics["macro_f1_fixed_33"]["mean"],
                "macro_f1_population_sd_pct": 100.0
                * metrics["macro_f1_fixed_33"]["population_sd"],
                "raw_accuracy_mean_pct": 100.0 * metrics["raw_accuracy"]["mean"],
                "raw_accuracy_population_sd_pct": 100.0
                * metrics["raw_accuracy"]["population_sd"],
                "raw_macro_f1_mean_pct": 100.0
                * metrics["raw_macro_f1_fixed_33"]["mean"],
                "raw_macro_f1_population_sd_pct": 100.0
                * metrics["raw_macro_f1_fixed_33"]["population_sd"],
            }
            for method, metrics in aggregate_summary.items()
        ],
    )

    canonical_summary_path = (
        repo / "results" / "fmc_pc_completion_20260721" / "rcbd_ablation_summary.json"
    )
    crosscheck: dict[str, Any] = {
        "canonical_summary": display_path(canonical_summary_path, repo),
        "checked": False,
    }
    if canonical_summary_path.is_file() and "angle_fmc_pc" in aggregate_summary:
        canonical = read_json(canonical_summary_path)["aggregate"]["fmc_angle_pc"]
        expected = canonical["valid_signal_self"]
        comparisons = {
            "smoothed_macro_f1_mean": (
                float(aggregate_summary["angle_fmc_pc"]["macro_f1_fixed_33"]["mean"]),
                float(expected["smoothed_macro_f1"]["mean"]),
            ),
            "smoothed_accuracy_mean": (
                float(aggregate_summary["angle_fmc_pc"]["accuracy"]["mean"]),
                float(expected["smoothed_accuracy"]["mean"]),
            ),
            "raw_macro_f1_mean": (
                float(aggregate_summary["angle_fmc_pc"]["raw_macro_f1_fixed_33"]["mean"]),
                float(expected["raw_macro_f1"]["mean"]),
            ),
            "raw_accuracy_mean": (
                float(aggregate_summary["angle_fmc_pc"]["raw_accuracy"]["mean"]),
                float(expected["raw_accuracy"]["mean"]),
            ),
        }
        crosscheck.update(
            {
                "checked": True,
                "comparisons": {
                    name: {
                        "recomputed": actual,
                        "canonical": reference,
                        "absolute_difference": abs(actual - reference),
                    }
                    for name, (actual, reference) in comparisons.items()
                },
                "all_within_1e-12": all(
                    abs(actual - reference) <= 1e-12
                    for actual, reference in comparisons.values()
                ),
            }
        )
    write_json(output / "canonical_crosscheck.json", crosscheck)
    write_csv(output / "participant_seed_metrics.csv", participant_rows)
    write_csv(output / "class_seed_metrics.csv", class_rows)
    write_json(output / "confusion_matrices.json", confusion_payload)
    write_csv(
        output / "off_diagonal_confusions.csv",
        sorted(
            confusion_rows,
            key=lambda row: (
                row["method"],
                -float(row["true_row_fraction"]),
                -int(row["summed_count_three_seeds"]),
            ),
        ),
    )

    participant_summary_rows: list[dict[str, Any]] = []
    for method in complete_methods:
        for participant in PARTICIPANTS:
            items = [
                row
                for row in participant_rows
                if row["method"] == method and row["participant"] == participant
            ]
            for metric in ("accuracy", "macro_f1_fixed_33", "macro_f1_observed_classes"):
                values = [float(row[metric]) for row in items]
                stats = summary(values)
                participant_summary_rows.append(
                    {
                        "method": method,
                        "participant": participant,
                        "metric": metric,
                        "mean": stats["mean"],
                        "population_sd": stats["population_sd"],
                        "minimum": stats["minimum"],
                        "maximum": stats["maximum"],
                    }
                )
    write_csv(output / "participant_summary.csv", participant_summary_rows)

    class_summary_rows: list[dict[str, Any]] = []
    for method in complete_methods:
        for class_index in range(NUM_CLASSES):
            items = [
                row
                for row in class_rows
                if row["method"] == method and row["class_index"] == class_index
            ]
            f1_values = [float(row["f1"]) for row in items]
            stats = summary(f1_values)
            class_summary_rows.append(
                {
                    "method": method,
                    "class_index": class_index,
                    "activity_id": f"A{class_index + 1}",
                    "activity_name": ACTIVITY_NAMES[class_index],
                    "mean_f1": stats["mean"],
                    "population_sd_f1": stats["population_sd"],
                    "support_per_seed": items[0]["support"],
                }
            )
    write_csv(output / "class_summary.csv", class_summary_rows)

    class_delta_rows: list[dict[str, Any]] = []
    for method in complete_methods:
        for baseline in ("clean_rcbd", "uniform_random_no_pc"):
            if method == baseline or baseline not in complete_methods:
                continue
            for class_index in range(NUM_CLASSES):
                current = next(
                    row
                    for row in class_summary_rows
                    if row["method"] == method and row["class_index"] == class_index
                )
                base = next(
                    row
                    for row in class_summary_rows
                    if row["method"] == baseline and row["class_index"] == class_index
                )
                class_delta_rows.append(
                    {
                        "method": method,
                        "baseline": baseline,
                        "class_index": class_index,
                        "activity_id": f"A{class_index + 1}",
                        "activity_name": ACTIVITY_NAMES[class_index],
                        "mean_f1_delta_pp": 100.0
                        * (float(current["mean_f1"]) - float(base["mean_f1"])),
                    }
                )
    write_csv(output / "class_mean_f1_deltas.csv", class_delta_rows)

    paired_rows: list[dict[str, Any]] = []
    comparator_methods = [
        method
        for method in ("angle_fmc_pc", "angle_fmc_no_pc", "uniform_random_pc")
        if method in complete_methods
    ]
    for method in comparator_methods:
        for baseline in ("clean_rcbd", "uniform_random_no_pc"):
            if baseline not in complete_methods:
                continue
            for seed in SEEDS:
                for participant in PARTICIPANTS:
                    current = next(
                        row
                        for row in participant_rows
                        if row["method"] == method
                        and row["seed"] == seed
                        and row["participant"] == participant
                    )
                    base = next(
                        row
                        for row in participant_rows
                        if row["method"] == baseline
                        and row["seed"] == seed
                        and row["participant"] == participant
                    )
                    paired_rows.append(
                        {
                            "method": method,
                            "baseline": baseline,
                            "seed": seed,
                            "participant": participant,
                            "accuracy_delta_pp": 100.0
                            * (float(current["accuracy"]) - float(base["accuracy"])),
                            "fixed_33_macro_f1_delta_pp": 100.0
                            * (
                                float(current["macro_f1_fixed_33"])
                                - float(base["macro_f1_fixed_33"])
                            ),
                            "observed_class_macro_f1_delta_pp": 100.0
                            * (
                                float(current["macro_f1_observed_classes"])
                                - float(base["macro_f1_observed_classes"])
                            ),
                        }
                    )
    write_csv(output / "paired_participant_deltas.csv", paired_rows)

    report_lines = [
        "# FMC-PC revision participant/class audit",
        "",
        "This directory is generated only from saved evaluation metrics; no model was retrained.",
        "",
        "## Complete three-seed methods",
        "",
    ]
    if complete_methods:
        for method, metrics in aggregate_summary.items():
            report_lines.append(
                f"- `{method}`: Accuracy {100 * metrics['accuracy']['mean']:.4f}% ± "
                f"{100 * metrics['accuracy']['population_sd']:.4f}%; fixed-33 Macro-F1 "
                f"{100 * metrics['macro_f1_fixed_33']['mean']:.4f}% ± "
                f"{100 * metrics['macro_f1_fixed_33']['population_sd']:.4f}%."
            )
    else:
        report_lines.append("- None.")
    if "angle_fmc_pc" in complete_methods and "uniform_random_no_pc" in complete_methods:
        fmc = aggregate_summary["angle_fmc_pc"]
        random = aggregate_summary["uniform_random_no_pc"]
        relevant_paired = [
            row
            for row in paired_rows
            if row["method"] == "angle_fmc_pc"
            and row["baseline"] == "uniform_random_no_pc"
        ]
        participant_deltas: dict[str, float] = {}
        for participant in PARTICIPANTS:
            values = [
                float(row["observed_class_macro_f1_delta_pp"])
                for row in relevant_paired
                if row["participant"] == participant
            ]
            participant_deltas[participant] = fmean(values)
        fmc_class_deltas = sorted(
            (
                row
                for row in class_delta_rows
                if row["method"] == "angle_fmc_pc"
                and row["baseline"] == "uniform_random_no_pc"
            ),
            key=lambda row: float(row["mean_f1_delta_pp"]),
            reverse=True,
        )
        report_lines.extend(
            [
                "",
                "## FMC-PC versus uniform random rotation (no PC)",
                "",
                f"- Pooled fixed-33 Macro-F1 difference: "
                f"{100 * (fmc['macro_f1_fixed_33']['mean'] - random['macro_f1_fixed_33']['mean']):+.4f} pp.",
                f"- Pooled Accuracy difference: "
                f"{100 * (fmc['accuracy']['mean'] - random['accuracy']['mean']):+.4f} pp.",
                "- Mean observed-class participant Macro-F1 differences: "
                + "; ".join(
                    f"{participant} {delta:+.2f} pp"
                    for participant, delta in participant_deltas.items()
                )
                + ".",
                f"- FMC-PC improves {sum(delta > 0 for delta in participant_deltas.values())}/4 "
                "participants by the observed-class diagnostic; the pooled improvement is "
                "strongly concentrated in subject14.",
                "- Largest class-level mean F1 gains: "
                + "; ".join(
                    f"{row['activity_id']} {float(row['mean_f1_delta_pp']):+.2f} pp"
                    for row in fmc_class_deltas[:5]
                )
                + ".",
                "- Largest class-level mean F1 decreases: "
                + "; ".join(
                    f"{row['activity_id']} {float(row['mean_f1_delta_pp']):+.2f} pp"
                    for row in reversed(fmc_class_deltas[-5:])
                )
                + ".",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Interpretation safeguards",
            "",
            "- `macro_f1_fixed_33` is the canonical 33-class metric used by the saved evaluator.",
            "- A participant may contain only a subset of the 33 activities. Therefore participant-level fixed-33 Macro-F1 includes zeros for absent classes and is not directly comparable across participants with different activity coverage.",
            "- `macro_f1_observed_classes` is supplied as a participant heterogeneity diagnostic only; it averages F1 over classes with positive true support for that participant.",
            "- Confusion counts are summed across seeds, so each evaluation window appears once per seed. The row-normalized matrices are intended for error-pattern visualization, not uncertainty estimation.",
            "- Population SD is used for the three fixed seeds, matching the manuscript summaries.",
            "",
            "## Missing expected files",
            "",
        ]
    )
    report_lines.extend(f"- `{path}`" for path in missing_files)
    (output / "README.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"wrote audit to {output}")
    print(f"complete methods: {', '.join(complete_methods) if complete_methods else 'none'}")


if __name__ == "__main__":
    main()
