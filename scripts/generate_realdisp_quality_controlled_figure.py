"""Generate the valid-signal and as-released REALDISP comparison figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = (
    "Clean RCBD",
    "Random rotation",
    "PD rotation",
    "PD rotation + gain",
    "PD full (no PC)",
    "Random rotation + PC",
    "PD-PC",
)
COLORS = {
    "Clean RCBD": "#6E6E6E",
    "Random rotation": "#0072B2",
    "PD rotation": "#56B4E9",
    "PD rotation + gain": "#009E73",
    "PD full (no PC)": "#E69F00",
    "Random rotation + PC": "#CC79A7",
    "PD-PC": "#D55E00",
}
MARKERS = {256: "o", 3407: "s", 114514: "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skill-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.skill_root.resolve() / "scripts"))
    from export_figure import export_figure  # noqa: PLC0415
    from layout_tools import add_panel_labels  # noqa: PLC0415
    from setup_style import setup_style  # noqa: PLC0415
    from visual_qa import audit_layout, print_report, render_preview  # noqa: PLC0415

    data = pd.read_csv(args.data.resolve())
    data = data.loc[data["scenario"] == "final_self"].copy()
    if len(data) != 21 or data["seed"].nunique() != 3:
        raise ValueError("Expected seven methods and three seeds for final Self")

    setup_style(journal="ieee", lang="en", use_sciplots=True)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 4.00),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    panels = (
        ("valid_signal_macro_f1", "Valid-signal cohort\n(subjects 14-17, n=4)"),
        ("as_released_macro_f1", "As-released cohort\n(subjects 13-17, n=5)"),
    )
    y_positions = np.arange(len(METHOD_ORDER))[::-1]

    for ax, (metric, title) in zip(axes, panels, strict=True):
        for y, method in zip(y_positions, METHOD_ORDER, strict=True):
            subset = data.loc[data["method_label"] == method].sort_values("seed")
            values = subset[metric].to_numpy(dtype=float)
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            ax.hlines(y, mean - std, mean + std, color=COLORS[method], linewidth=1.2)
            ax.plot(
                [mean],
                [y],
                marker="D",
                markersize=5.4,
                color=COLORS[method],
                markeredgecolor="black",
                markeredgewidth=0.45,
                linestyle="none",
                zorder=3,
            )
            offsets = (-0.16, 0.0, 0.16)
            for offset, (_, row) in zip(offsets, subset.iterrows(), strict=True):
                ax.plot(
                    row[metric],
                    y + offset,
                    marker=MARKERS[int(row["seed"])],
                    markersize=4.2,
                    markerfacecolor="white",
                    markeredgecolor=COLORS[method],
                    markeredgewidth=0.9,
                    linestyle="none",
                    zorder=4,
                )
            if method == "PD-PC":
                ax.text(
                    min(mean + std + 1.5, 96.5),
                    y,
                    f"{mean:.2f}",
                    va="center",
                    ha="left",
                    fontsize=8,
                    fontweight="bold",
                    color=COLORS[method],
                )
        ax.set_title(title, fontsize=9, pad=8)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Self-placement Macro-F1 (%)")
        ax.set_xticks(np.arange(0, 101, 20))
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.5, linestyle="--", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(METHOD_ORDER)

    add_panel_labels(fig, style="ieee")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preview = output_dir / "_preview_quality_controlled_ablation.png"
    render_preview(fig, preview, dpi=150)
    issues = audit_layout(fig)
    print_report(issues)
    if any(item.get("level") == "FAIL" for item in issues):
        raise RuntimeError("Figure layout audit failed")
    export_figure(
        fig,
        str(output_dir / "figure2_quality_controlled_ablation"),
        formats=["pdf", "svg", "png"],
        size_inches=(7.16, 4.00),
        dpi=600,
        grayscale_preview=True,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
