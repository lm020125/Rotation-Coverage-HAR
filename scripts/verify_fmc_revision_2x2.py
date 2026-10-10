from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any


SEEDS = (256, 3407, 114514)
METHOD_TO_CELL = {
    "uniform_random_no_pc": "uniform_no_pc",
    "uniform_random_pc": "uniform_pc",
    "angle_fmc_no_pc": "empirical_no_pc",
    "angle_fmc_pc": "empirical_pc",
}
TOLERANCE = 1e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def effect_values(cells: dict[int, dict[str, float]]) -> dict[str, list[float]]:
    empirical_main = []
    pc_main = []
    interaction = []
    for seed in SEEDS:
        c = cells[seed]
        empirical_main.append(
            0.5
            * (
                (c["empirical_no_pc"] - c["uniform_no_pc"])
                + (c["empirical_pc"] - c["uniform_pc"])
            )
        )
        pc_main.append(
            0.5
            * (
                (c["uniform_pc"] - c["uniform_no_pc"])
                + (c["empirical_pc"] - c["empirical_no_pc"])
            )
        )
        interaction.append(
            (c["empirical_pc"] - c["empirical_no_pc"])
            - (c["uniform_pc"] - c["uniform_no_pc"])
        )
    return {
        "empirical_amplitude_main_effect": empirical_main,
        "prediction_consistency_main_effect": pc_main,
        "amplitude_x_pc_interaction": interaction,
    }


def direct_index(
    rows: list[dict[str, str]], keys: tuple[str, ...]
) -> dict[tuple[str, ...], dict[str, str]]:
    indexed: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        if key in indexed:
            raise ValueError(f"Duplicate key {key}")
        indexed[key] = row
    return indexed


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    result_root = (
        repo / "results" / "fmc_pc_revision_20260723_local"
    )
    final_root = result_root / "participant_class_final"
    effects_path = result_root / "server_results" / "effects_2x2.json"
    aggregate_path = final_root / "aggregate_seed_metrics.csv"
    participant_path = final_root / "participant_seed_metrics.csv"
    class_path = final_root / "class_seed_metrics.csv"
    paired_path = final_root / "paired_participant_deltas.csv"
    class_delta_path = final_root / "class_mean_f1_deltas.csv"
    output_root = result_root / "figure_revision_2x2"
    output_root.mkdir(parents=True, exist_ok=True)

    aggregate_rows = read_csv(aggregate_path)
    effects_payload = read_json(effects_path)
    aggregate_index = direct_index(aggregate_rows, ("method", "seed"))

    cells: dict[int, dict[str, float]] = {seed: {} for seed in SEEDS}
    checks: list[dict[str, Any]] = []
    for method, cell in METHOD_TO_CELL.items():
        for seed in SEEDS:
            actual = float(aggregate_index[(method, str(seed))]["macro_f1_fixed_33"])
            expected = float(
                effects_payload["metrics"]["macro_f1"]["cell_values_by_seed"][str(seed)][cell]
            )
            difference = abs(actual - expected)
            cells[seed][cell] = actual
            checks.append(
                {
                    "scope": "2x2_cell",
                    "method": method,
                    "cell": cell,
                    "seed": seed,
                    "actual": actual,
                    "expected": expected,
                    "absolute_difference": difference,
                    "pass": difference <= TOLERANCE,
                }
            )

    recomputed_effects = effect_values(cells)
    effect_audit: dict[str, Any] = {}
    for name, values in recomputed_effects.items():
        expected = effects_payload["metrics"]["macro_f1"]["effects"][name]
        expected_values = [float(value) for value in expected["values"]]
        value_differences = [abs(a - b) for a, b in zip(values, expected_values)]
        mean_value = fmean(values)
        sd_value = pstdev(values)
        mean_difference = abs(mean_value - float(expected["mean"]))
        sd_difference = abs(sd_value - float(expected["std_population"]))
        effect_audit[name] = {
            "values": values,
            "values_percentage_points": [100.0 * value for value in values],
            "mean": mean_value,
            "mean_percentage_points": 100.0 * mean_value,
            "population_sd": sd_value,
            "population_sd_percentage_points": 100.0 * sd_value,
            "max_cell_value_difference": max(value_differences),
            "mean_difference": mean_difference,
            "population_sd_difference": sd_difference,
            "pass": max(value_differences + [mean_difference, sd_difference]) <= TOLERANCE,
        }

    participant_rows = read_csv(participant_path)
    class_rows = read_csv(class_path)
    paired_rows = read_csv(paired_path)
    class_delta_rows = read_csv(class_delta_path)
    participant_index = direct_index(participant_rows, ("method", "seed", "participant"))
    class_index = direct_index(class_rows, ("method", "seed", "class_index"))

    expected_participant_keys = 5 * 3 * 4
    expected_class_keys = 5 * 3 * 33
    structure_checks = {
        "participant_rows": {
            "actual": len(participant_rows),
            "expected": expected_participant_keys,
            "pass": len(participant_rows) == expected_participant_keys,
        },
        "class_rows": {
            "actual": len(class_rows),
            "expected": expected_class_keys,
            "pass": len(class_rows) == expected_class_keys,
        },
    }

    participant_delta_checks = []
    relevant_paired = {
        (row["seed"], row["participant"]): row
        for row in paired_rows
        if row["method"] == "angle_fmc_pc"
        and row["baseline"] == "uniform_random_no_pc"
    }
    for seed in SEEDS:
        for participant in ("subject14", "subject15", "subject16", "subject17"):
            direct_delta = 100.0 * (
                float(participant_index[("angle_fmc_pc", str(seed), participant)]["macro_f1_fixed_33"])
                - float(
                    participant_index[("uniform_random_no_pc", str(seed), participant)][
                        "macro_f1_fixed_33"
                    ]
                )
            )
            recorded_delta = float(
                relevant_paired[(str(seed), participant)]["fixed_33_macro_f1_delta_pp"]
            )
            difference = abs(direct_delta - recorded_delta)
            participant_delta_checks.append(
                {
                    "seed": seed,
                    "participant": participant,
                    "direct_delta_pp": direct_delta,
                    "recorded_delta_pp": recorded_delta,
                    "absolute_difference": difference,
                    "pass": difference <= TOLERANCE,
                }
            )

    class_delta_index = {
        int(row["class_index"]): row
        for row in class_delta_rows
        if row["method"] == "angle_fmc_pc"
        and row["baseline"] == "uniform_random_no_pc"
    }
    class_delta_checks = []
    for activity in range(33):
        deltas = [
            100.0
            * (
                float(class_index[("angle_fmc_pc", str(seed), str(activity))]["f1"])
                - float(class_index[("uniform_random_no_pc", str(seed), str(activity))]["f1"])
            )
            for seed in SEEDS
        ]
        direct_delta = fmean(deltas)
        recorded_delta = float(class_delta_index[activity]["mean_f1_delta_pp"])
        difference = abs(direct_delta - recorded_delta)
        class_delta_checks.append(
            {
                "class_index": activity,
                "activity_id": f"A{activity + 1}",
                "seed_deltas_pp": deltas,
                "direct_mean_delta_pp": direct_delta,
                "recorded_mean_delta_pp": recorded_delta,
                "absolute_difference": difference,
                "pass": difference <= TOLERANCE,
            }
        )

    all_pass = (
        all(item["pass"] for item in checks)
        and all(item["pass"] for item in effect_audit.values())
        and all(item["pass"] for item in structure_checks.values())
        and all(item["pass"] for item in participant_delta_checks)
        and all(item["pass"] for item in class_delta_checks)
    )
    output = {
        "status": "pass" if all_pass else "fail",
        "tolerance": TOLERANCE,
        "input_sha256": {
            path.name: sha256(path)
            for path in (
                aggregate_path,
                participant_path,
                class_path,
                paired_path,
                class_delta_path,
                effects_path,
            )
        },
        "cell_checks": checks,
        "effect_audit": effect_audit,
        "structure_checks": structure_checks,
        "participant_delta_checks": participant_delta_checks,
        "class_delta_checks": class_delta_checks,
        "guardrail": (
            "The three fixed seeds are descriptive computational replicates. "
            "No p-values or confirmatory population-level significance claims are supported."
        ),
    }
    output_path = output_root / "independent_verification.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    lines = [
        "# Independent FMC-PC 2x2 verification",
        "",
        f"Overall status: **{output['status'].upper()}**.",
        "",
        "## Macro-F1 factorial effects",
        "",
        "| Effect | Seed values (pp) | Mean ± population SD (pp) |",
        "|---|---:|---:|",
    ]
    labels = {
        "empirical_amplitude_main_effect": "Empirical amplitude main effect",
        "prediction_consistency_main_effect": "Prediction-consistency main effect",
        "amplitude_x_pc_interaction": "Amplitude × PC interaction",
    }
    for name, result in effect_audit.items():
        values = ", ".join(f"{value:.2f}" for value in result["values_percentage_points"])
        lines.append(
            f"| {labels[name]} | {values} | "
            f"{result['mean_percentage_points']:.2f} ± "
            f"{result['population_sd_percentage_points']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- All four 2x2 cells match the saved aggregate results within 1e-12.",
            "- Participant rows (5 methods × 3 seeds × 4 participants) and class rows "
            "(5 methods × 3 seeds × 33 classes) are complete.",
            "- Direct participant and class differences match the saved delta tables within 1e-12.",
            "- n=3 seeds is descriptive only; no significance test or population-level confidence claim is made.",
            "",
        ]
    )
    (output_root / "independent_verification.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    if not all_pass:
        raise SystemExit("Independent verification failed; inspect independent_verification.json")
    print(f"PASS: wrote {output_path}")


if __name__ == "__main__":
    main()
