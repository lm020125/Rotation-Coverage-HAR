"""Generate auditable REALDISP manuscript tables from frozen paper data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


MAIN_METHODS = {
    "clean_rcbd": "Clean",
    "random_rotation": "Random rotation",
    "pd_pc": "PD-PC",
    "dcl_clean": "Clean",
    "dcl_random_rotation": "Random rotation",
    "dcl_pd_pc": "PD-PC",
}
COMPARISON_LABELS = {
    "random_rotation_vs_clean_rcbd": "Random rotation vs Clean",
    "pd_pc_vs_clean_rcbd": "PD-PC vs Clean",
    "pd_pc_vs_random_rotation": "PD-PC vs Random rotation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--realdisp-root",
        type=Path,
        required=True,
        help="REALDISP archive directory containing outputs/ and paper_assets/data/.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_sd(row: pd.Series, metric: str) -> str:
    return f"{row[f'{metric}_mean']:.2f} ± {row[f'{metric}_std']:.2f}"


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "±": r"$\pm$",
        "–": "--",
        "—": "--",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def latex_table(frame: pd.DataFrame, caption: str, label: str) -> str:
    columns = "l" + "c" * (len(frame.columns) - 1)
    header = " & ".join(latex_escape(column) for column in frame.columns) + r" \\"
    rows = [
        " & ".join(latex_escape(value) for value in row) + r" \\"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            rf"\caption{{{latex_escape(caption)}}}",
            rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{columns}}}",
            r"\toprule",
            header,
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )


def table1_protocol() -> pd.DataFrame:
    rows = [
        ("Supervised training", "subjects 1–10, Ideal labels"),
        ("Placement calibration", "subjects 1–10, unlabeled Self sensor statistics"),
        ("Validation", "subjects 11–12, Ideal and Self"),
        ("Frozen final test", "subjects 13–17, Ideal and Self"),
        ("Random seeds", "256, 3407, 114514"),
        (
            "Paired headline estimands",
            "as-released and post hoc valid-signal sample-pooled Macro-F1",
        ),
        ("Secondary metric", "Accuracy"),
        ("Reported uncertainty", "Mean ± SD over three seeds"),
        ("Statistical unit", "Participant; seeds averaged within participant"),
        (
            "Frozen as-released analysis",
            "subjects13–17; retains the subject13 all-zero Self record",
        ),
        (
            "Post hoc valid-signal conditional analysis",
            "subjects14–17; label-free raw-input rule; no retraining",
        ),
    ]
    return pd.DataFrame(rows, columns=["Protocol item", "Frozen specification"])


def table2_main(ablation: pd.DataFrame, dcl: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for backbone, source, methods in (
        ("RCBD", ablation, ("clean_rcbd", "random_rotation", "pd_pc")),
        ("DeepConvLSTM", dcl, ("dcl_clean", "dcl_random_rotation", "dcl_pd_pc")),
    ):
        selected = source.set_index("method")
        clean_self = float(selected.loc[methods[0], "final_self_macro_f1_mean"])
        for method in methods:
            row = selected.loc[method]
            rows.append(
                {
                    "Backbone": backbone,
                    "Training condition": MAIN_METHODS[method],
                    "Ideal Macro-F1 (%)": mean_sd(row, "final_ideal_macro_f1"),
                    "Self Macro-F1 (%)": mean_sd(row, "final_self_macro_f1"),
                    "Self Accuracy (%)": mean_sd(row, "final_self_accuracy"),
                    "ΔSelf vs Clean (pp)": (
                        "—"
                        if method == methods[0]
                        else f"{row['final_self_macro_f1_mean'] - clean_self:+.2f}"
                    ),
                }
            )
    return pd.DataFrame(rows)


def table3_ablation(ablation: pd.DataFrame) -> pd.DataFrame:
    clean_self = float(
        ablation.loc[ablation["method"] == "clean_rcbd", "final_self_macro_f1_mean"].iloc[0]
    )
    rows = []
    for _, row in ablation.sort_values("method_order").iterrows():
        rows.append(
            {
                "Method": row["method_label"],
                "Ideal Macro-F1 (%)": mean_sd(row, "final_ideal_macro_f1"),
                "Self Macro-F1 (%)": mean_sd(row, "final_self_macro_f1"),
                "Self Accuracy (%)": mean_sd(row, "final_self_accuracy"),
                "ΔSelf vs Clean (pp)": (
                    "—"
                    if row["method"] == "clean_rcbd"
                    else f"{row['final_self_macro_f1_mean'] - clean_self:+.2f}"
                ),
            }
        )
    return pd.DataFrame(rows)


def table4_statistics(statistics: dict[str, Any]) -> pd.DataFrame:
    comparisons = statistics["scenarios"]["final_self_subjects13_17"]["comparisons"]
    rows = []
    for key in COMPARISON_LABELS:
        payload = comparisons[key]
        pooled = payload["descriptive_pooled_macro_f1_difference"]
        ci_low, ci_high = payload["bootstrap_95_ci"]
        rows.append(
            {
                "Comparison": COMPARISON_LABELS[key],
                "Pooled Δ (pp)": f"{pooled['mean'] * 100:+.2f}",
                "Participant mean Δ (pp)": f"{payload['mean_difference'] * 100:+.2f}",
                "Participant bootstrap 95% CI (pp)": (
                    f"[{ci_low * 100:+.2f}, {ci_high * 100:+.2f}]"
                ),
                "Exact sign-flip p": f"{payload['exact_two_sided_sign_flip_pvalue']:.3f}",
                "Paired dz": f"{payload['paired_cohens_dz']:.3f}",
                "Improved / harmed / unchanged": (
                    f"{payload['improved_participants']} / "
                    f"{payload['harmed_participants']} / "
                    f"{payload['unchanged_participants']}"
                ),
            }
        )
    return pd.DataFrame(rows)


def write_table_set(
    output: Path,
    stem: str,
    frame: pd.DataFrame,
    caption: str,
    label: str,
) -> tuple[str, str]:
    frame.to_csv(output / f"{stem}.csv", index=False, encoding="utf-8-sig")
    markdown = f"## {caption}\n\n{markdown_table(frame)}\n"
    latex = latex_table(frame, caption, label)
    (output / f"{stem}.md").write_text(markdown, encoding="utf-8")
    (output / f"{stem}.tex").write_text(latex + "\n", encoding="utf-8")
    return markdown, latex


def main() -> None:
    args = parse_args()
    realdisp_root = args.realdisp_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = realdisp_root / "paper_assets" / "data"
    ablation_path = data / "ablation_summary.csv"
    dcl_path = data / "deepconvlstm_summary.csv"
    statistics_path = (
        realdisp_root
        / "outputs"
        / "realdisp_pd_pc_statistical_audit_20260719"
        / "statistical_analysis.json"
    )
    inputs = [ablation_path, dcl_path, statistics_path]
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen inputs: {missing}")

    ablation = pd.read_csv(ablation_path)
    dcl = pd.read_csv(dcl_path)
    with statistics_path.open("r", encoding="utf-8") as handle:
        statistics = json.load(handle)

    tables = [
        (
            "table1_protocol",
            table1_protocol(),
            "Table 1. Frozen REALDISP protocol",
            "tab:protocol",
        ),
        (
            "table2_main_results",
            table2_main(ablation, dcl),
            "Table 2. Main results and cross-backbone evaluation",
            "tab:main-results",
        ),
        (
            "table3_component_ablation",
            table3_ablation(ablation),
            "Table 3. Predefined component ablation on the frozen final test",
            "tab:ablation",
        ),
        (
            "table4_participant_statistics",
            table4_statistics(statistics),
            "Table 4. Participant-level inference on Self-placement",
            "tab:participant-statistics",
        ),
    ]
    markdown_parts = [
        "# REALDISP manuscript tables\n",
        "All performance values are percentages. Mean ± SD is computed over three seeds "
        "(256, 3407, 114514). Statistical inference averages seeds within each participant "
        "before analysis (n=5 participants). subject13 is retained in the main results "
        "despite its all-zero four-limb Self record.\n",
    ]
    latex_parts = [
        "% Auto-generated REALDISP manuscript tables.",
        "% Requires \\usepackage{booktabs}.\n",
    ]
    for stem, frame, caption, label in tables:
        markdown, latex = write_table_set(output, stem, frame, caption, label)
        markdown_parts.append(markdown)
        latex_parts.append(latex + "\n")
    (output / "PAPER_TABLES.md").write_text("\n".join(markdown_parts), encoding="utf-8")
    (output / "PAPER_TABLES.tex").write_text("\n".join(latex_parts), encoding="utf-8")

    generated = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "table_manifest.json"
    )
    manifest = {
        "purpose": "auto-generated manuscript tables from frozen REALDISP results",
        "inputs": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in inputs
        ],
        "outputs": [
            {"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in generated
        ],
        "table_count": len(tables),
        "figures_created": 0,
    }
    with (output / "table_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(output), "tables": len(tables)}, indent=2))


if __name__ == "__main__":
    main()
