"""Generate paired frozen and post hoc conditional REALDISP paper tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MAIN_METHODS = ("clean_rcbd", "random_rotation", "pd_pc")
DCL_LABELS = {
    "dcl_clean": "Clean",
    "dcl_random_rotation": "Random rotation",
    "dcl_pd_pc": "PD-PC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def fmt(mean: float, std: float) -> str:
    return f"{mean:.2f} ± {std:.2f}"


def write_table(frame: pd.DataFrame, stem: str, output: Path) -> None:
    frame.to_csv(output / f"{stem}.csv", index=False, encoding="utf-8-sig")
    headers = [str(value) for value in frame.columns]
    markdown_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        markdown_lines.append(
            "| " + " | ".join(str(value) for value in row) + " |"
        )
    (output / f"{stem}.md").write_text(
        "\n".join(markdown_lines) + "\n", encoding="utf-8"
    )
    def latex_escape(value: object) -> str:
        text = str(value)
        for source, target in (
            ("\\", r"\textbackslash{}"),
            ("&", r"\&"),
            ("%", r"\%"),
            ("_", r"\_"),
            ("Δ", r"$\Delta$"),
            ("±", r"$\pm$"),
            ("—", "--"),
        ):
            text = text.replace(source, target)
        return text

    alignment = "l" * len(headers)
    latex_lines = [r"\begin{tabular}{" + alignment + "}", r"\toprule"]
    latex_lines.append(" & ".join(latex_escape(value) for value in headers) + r" \\")
    latex_lines.append(r"\midrule")
    for row in frame.itertuples(index=False, name=None):
        latex_lines.append(
            " & ".join(latex_escape(value) for value in row) + r" \\"
        )
    latex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output / f"{stem}.tex").write_text(
        "\n".join(latex_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    data = args.data_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(data / "quality_controlled_summary.csv")
    dcl = pd.read_csv(data / "quality_controlled_deepconvlstm_summary.csv")
    comparisons = pd.read_csv(data / "quality_controlled_participant_comparisons.csv")
    thresholds = pd.read_csv(data / "quality_threshold_sensitivity.csv")

    rcbd_rows = []
    for method in MAIN_METHODS:
        ideal = summary.loc[
            (summary["scenario"] == "final_ideal") & (summary["method"] == method)
        ].iloc[0]
        self_row = summary.loc[
            (summary["scenario"] == "final_self") & (summary["method"] == method)
        ].iloc[0]
        rcbd_rows.append(
            {
                "Backbone": "RCBD",
                "Condition": self_row["method_label"].replace(" RCBD", ""),
                "Valid Ideal Macro-F1 (%)": fmt(
                    ideal["valid_signal_macro_f1_mean"],
                    ideal["valid_signal_macro_f1_std"],
                ),
                "Valid Self Macro-F1 (%)": fmt(
                    self_row["valid_signal_macro_f1_mean"],
                    self_row["valid_signal_macro_f1_std"],
                ),
                "As-released Self Macro-F1 (%)": fmt(
                    self_row["as_released_macro_f1_mean"],
                    self_row["as_released_macro_f1_std"],
                ),
            }
        )
    dcl_rows = []
    for method, label in DCL_LABELS.items():
        ideal = dcl.loc[
            (dcl["scenario"] == "final_ideal") & (dcl["method"] == method)
        ].iloc[0]
        self_row = dcl.loc[
            (dcl["scenario"] == "final_self") & (dcl["method"] == method)
        ].iloc[0]
        dcl_rows.append(
            {
                "Backbone": "DeepConvLSTM",
                "Condition": label,
                "Valid Ideal Macro-F1 (%)": fmt(
                    ideal["valid_signal_macro_f1_mean"],
                    ideal["valid_signal_macro_f1_std"],
                ),
                "Valid Self Macro-F1 (%)": fmt(
                    self_row["valid_signal_macro_f1_mean"],
                    self_row["valid_signal_macro_f1_std"],
                ),
                "As-released Self Macro-F1 (%)": fmt(
                    self_row["as_released_macro_f1_mean"],
                    self_row["as_released_macro_f1_std"],
                ),
            }
        )
    write_table(
        pd.DataFrame(rcbd_rows + dcl_rows),
        "table2_quality_controlled_main",
        output,
    )

    ablation_rows = []
    clean_self = summary.loc[
        (summary["scenario"] == "final_self") & (summary["method"] == "clean_rcbd")
    ].iloc[0]["valid_signal_macro_f1_mean"]
    for method in summary.loc[summary["scenario"] == "final_self", "method"]:
        ideal = summary.loc[
            (summary["scenario"] == "final_ideal") & (summary["method"] == method)
        ].iloc[0]
        self_row = summary.loc[
            (summary["scenario"] == "final_self") & (summary["method"] == method)
        ].iloc[0]
        ablation_rows.append(
            {
                "Method": self_row["method_label"],
                "Valid Ideal Macro-F1 (%)": fmt(
                    ideal["valid_signal_macro_f1_mean"],
                    ideal["valid_signal_macro_f1_std"],
                ),
                "Valid Self Macro-F1 (%)": fmt(
                    self_row["valid_signal_macro_f1_mean"],
                    self_row["valid_signal_macro_f1_std"],
                ),
                "As-released Self Macro-F1 (%)": fmt(
                    self_row["as_released_macro_f1_mean"],
                    self_row["as_released_macro_f1_std"],
                ),
                "Valid ΔSelf vs Clean (pp)": (
                    "—"
                    if method == "clean_rcbd"
                    else f"{self_row['valid_signal_macro_f1_mean'] - clean_self:+.2f}"
                ),
            }
        )
    write_table(
        pd.DataFrame(ablation_rows),
        "table3_quality_controlled_ablation",
        output,
    )

    comparison_table = comparisons.rename(
        columns={
            "comparison": "Comparison",
            "pooled_macro_f1_delta_pp": "Pooled Δ (pp)",
            "participant_equal_mean_delta_pp": "Participant mean Δ (pp)",
            "bootstrap_95_ci_low_pp": "CI low (pp)",
            "bootstrap_95_ci_high_pp": "CI high (pp)",
            "exact_two_sided_sign_flip_p": "Exact sign-flip p",
            "paired_cohens_dz": "Paired dz",
            "improved": "Improved",
            "harmed": "Harmed",
            "unchanged": "Unchanged",
        }
    )[
        [
            "Comparison",
            "Pooled Δ (pp)",
            "Participant mean Δ (pp)",
            "CI low (pp)",
            "CI high (pp)",
            "Exact sign-flip p",
            "Paired dz",
            "Improved",
            "Harmed",
            "Unchanged",
        ]
    ]
    write_table(comparison_table, "table4_quality_controlled_participant", output)

    threshold_table = thresholds.rename(
        columns={
            "per_stream_zero_fraction_threshold": "Zero-fraction threshold",
            "excluded_participants": "Excluded",
            "included_participants": "Included",
            "included_n": "n",
            "pd_pc_self_macro_f1_mean": "PD-PC Self Macro-F1 mean (%)",
            "pd_pc_self_macro_f1_std": "SD (%)",
        }
    )[
        [
            "Zero-fraction threshold",
            "Excluded",
            "Included",
            "n",
            "PD-PC Self Macro-F1 mean (%)",
            "SD (%)",
        ]
    ]
    write_table(threshold_table, "table5_quality_threshold_sensitivity", output)


if __name__ == "__main__":
    main()
