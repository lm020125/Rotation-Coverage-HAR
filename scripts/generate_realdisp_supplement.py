from __future__ import annotations

import argparse
import csv
from pathlib import Path


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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value: str | float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the REALDISP PD-PC supplement.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.data_dir.resolve()

    ablation = read_rows(data / "ablation_runs.csv")
    participants = read_rows(data / "participant_self_summary.csv")
    classes = read_rows(data / "class_self_main.csv")
    sensitivity = read_rows(data / "sensitivity_runs.csv")

    sections: list[str] = [
        "# Supplementary material",
        "",
        "## S1. Frozen experimental protocol",
        "",
        "The protocol schematic is reproduced below. Subjects 1–10 provide labelled "
        "Ideal training data and unlabeled Self-placement sensor statistics; subjects "
        "11–12 are used for validation; and subjects 13–17 are accessed only for the "
        "frozen final evaluation. No final-participant signal contributes to profile "
        "estimation, normalization, early stopping, or hyperparameter selection.",
        "",
        "![Figure S4. Frozen participant and data-access protocol.](../figures/figureS4_experimental_protocol.pdf)",
        "",
        "## S2. Exact seed-level RCBD ablation",
        "",
    ]

    rows = []
    for item in sorted(ablation, key=lambda x: (int(x["method_order"]), int(x["seed"]))):
        rows.append(
            [
                item["method_label"],
                item["seed"],
                fmt(item["final_ideal_macro_f1"]),
                fmt(item["final_self_macro_f1"]),
                fmt(item["final_self_accuracy"]),
            ]
        )
    sections.extend(
        [
            md_table(
                ["Method", "Seed", "Ideal Macro-F1 (%)", "Self Macro-F1 (%)", "Self Accuracy (%)"],
                rows,
            ),
            "",
            "![Figure S3. Seed-level Self-placement Accuracy for all RCBD ablations.](../figures/figureS3_self_accuracy.pdf)",
            "",
            "## S3. Participant-level Self-placement results",
            "",
            "Subject 13 is retained in the frozen as-released analysis. Its four distal-limb "
            "Self acceleration arrays are all zero, and all methods therefore yield the same "
            "near-zero score. The valid-signal estimate excluding this record is a post hoc "
            "conditional analysis of the same frozen predictions. Values below are Macro-F1 percentages.",
            "",
        ]
    )

    rows = []
    for item in sorted(
        participants,
        key=lambda x: (int(x["participant"].replace("subject", "")), int(x["method_order"])),
    ):
        rows.append(
            [
                item["participant"],
                item["data_status"],
                item["method_label"],
                fmt(item["seed_256"]),
                fmt(item["seed_3407"]),
                fmt(item["seed_114514"]),
                f"{fmt(item['mean'])} ± {fmt(item['std'])}",
            ]
        )
    sections.extend(
        [
            md_table(
                ["Participant", "Record status", "Method", "Seed 256", "Seed 3407", "Seed 114514", "Mean ± SD"],
                rows,
            ),
            "",
            "## S4. Complete class-wise results",
            "",
            "![Figure S1. Complete class-wise Self Macro-F1 and differences.](../figures/figureS1_classwise_differences.pdf)",
            "",
        ]
    )

    by_class: dict[int, dict[str, dict[str, str]]] = {}
    for item in classes:
        by_class.setdefault(int(item["class_index"]), {})[item["method"]] = item
    rows = []
    for index in range(33):
        methods = by_class[index]
        clean = methods["clean_rcbd"]
        rotation = methods["random_rotation"]
        pd_pc = methods["pd_pc"]
        delta = float(pd_pc["mean_f1"]) - float(clean["mean_f1"])
        rows.append(
            [
                f"A{index + 1}",
                ACTIVITY_NAMES[index],
                f"{fmt(clean['mean_f1'])} ± {fmt(clean['std_f1'])}",
                f"{fmt(rotation['mean_f1'])} ± {fmt(rotation['std_f1'])}",
                f"{fmt(pd_pc['mean_f1'])} ± {fmt(pd_pc['std_f1'])}",
                f"{delta:+.2f}",
            ]
        )
    sections.extend(
        [
            md_table(
                ["ID", "Activity", "Clean RCBD", "Random rotation", "PD-PC", "PD-PC − Clean (pp)"],
                rows,
            ),
            "",
            "## S5. Validation-only hyperparameter sensitivity",
            "",
            "These runs were performed after the central configuration had been frozen. "
            "They use only subjects 1–10 for training and subjects 11–12 for evaluation; "
            "subjects 13–17 were not accessed.",
            "",
        ]
    )

    rows = []
    for item in sorted(sensitivity, key=lambda x: (int(x["setting_order"]), int(x["seed"]))):
        rows.append(
            [
                item["setting"],
                fmt(item["augmentation_probability"]),
                fmt(item["consistency_weight"]),
                item["seed"],
                fmt(item["validation_ideal_macro_f1"]),
                fmt(item["validation_rotation_stress_macro_f1"]),
                fmt(item["validation_self_macro_f1"]),
            ]
        )
    sections.extend(
        [
            md_table(
                ["Setting", "p", "lambda", "Seed", "Ideal Macro-F1 (%)", "Rotation-stress Macro-F1 (%)", "Self Macro-F1 (%)"],
                rows,
            ),
            "",
            "![Figure S2. Ideal and rotation-stress sensitivity on the validation participants.](../figures/figureS2_sensitivity_other_scenarios.pdf)",
            "",
            "## S6. Data-quality audit and sensitivity",
            "",
            "The profile estimator rejects a limb-domain record when more than 50% of "
            "acceleration-vector norms are at most 1e-6. This rule rejected all four Self "
            "limbs of subject 6 from the training profile bank. The processed Self record "
            "for final-test subject 13 was also all zero and remains in the frozen as-released "
            "evaluation because the final cohort had already been fixed.",
            "",
            "Frozen five-participant pooled Self Macro-F1 is 66.4994% for the full "
            "calibrated perturbation without prediction consistency and 68.7472% for "
            "PD-PC. Excluding subject 13 only as a sensitivity analysis gives 78.1466% "
            "and 81.1025%, respectively. This post hoc conditional estimate is reported "
            "alongside, not substituted for, the frozen as-released benchmark.",
            "",
            "## S7. Reproducibility inventory",
            "",
            "- Data preparation: `scripts/prepare_realdisp.py`.",
            "- Label-free profiles: `scripts/estimate_realdisp_placement_profiles.py`.",
            "- Frozen component ablation: `scripts/run_realdisp_pd_pc_component_ablation.py`.",
            "- Validation sensitivity: `scripts/run_realdisp_pd_pc_sensitivity.py`.",
            "- Statistical audit: `scripts/analyze_realdisp_pd_pc_participants.py`.",
            "- Paper data, tables, and figures: `scripts/prepare_realdisp_paper_data.py`, "
            "`scripts/generate_realdisp_paper_tables.py`, "
            "`scripts/generate_realdisp_paper_figures.py`, and "
            "`scripts/generate_realdisp_method_diagrams.py`.",
            "- Machine-readable source hashes: `paper_assets/data/paper_result_catalog.json`.",
            "",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sections), encoding="utf-8")
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
