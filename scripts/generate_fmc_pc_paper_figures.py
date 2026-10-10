from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


SKILL_SCRIPTS = Path.home() / ".codex" / "skills" / "scipilot-figure-skill" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))

from export_figure import export_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402


COLORS = {
    "grey": "#6B7280",
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#7B61A8",
}
SEED_MARKERS = {256: "o", 3407: "s", 114514: "^"}


def _style_axis(ax: plt.Axes, ylabel: str, ylim: tuple[float, float]) -> None:
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _dot_summary(
    ax: plt.Axes,
    frame: pd.DataFrame,
    order: list[str],
    value: str,
    colors: list[str],
    labels: list[str],
) -> None:
    for i, (method, color) in enumerate(zip(order, colors, strict=True)):
        part = frame.loc[frame["method"] == method].sort_values("seed")
        vals = part[value].to_numpy(float)
        seeds = part["seed"].to_numpy(int)
        offsets = np.linspace(-0.11, 0.11, len(vals))
        for off, seed, val in zip(offsets, seeds, vals, strict=True):
            ax.scatter(
                i + off,
                val,
                marker=SEED_MARKERS[int(seed)],
                s=30,
                facecolor="white",
                edgecolor=color,
                linewidth=1.0,
                zorder=3,
            )
        mean = float(vals.mean())
        sd = float(vals.std(ddof=0))
        ax.errorbar(
            i,
            mean,
            yerr=sd,
            fmt="D",
            ms=4.2,
            color=color,
            ecolor=color,
            capsize=3,
            elinewidth=1.1,
            zorder=4,
        )
    ax.set_xticks(range(len(order)), labels, rotation=18, ha="right")


def figure_method(out_dir: Path) -> None:
    setup_style(journal="ieee", lang="en", use_sciplots=False)
    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.03, 0.59, 0.19, 0.25, "Train participants\nIdeal + unlabeled Self", COLORS["grey"]),
        (0.29, 0.59, 0.21, 0.25, "Estimate empirical\nIdeal-Self angle marginal", COLORS["blue"]),
        (0.57, 0.59, 0.19, 0.25, "Use as rotation magnitude;\nrandomize 3-D axis", COLORS["orange"]),
        (0.81, 0.59, 0.16, 0.25, "One-side\nrotation", COLORS["green"]),
        (0.22, 0.12, 0.22, 0.24, "Clean view\n$ f_\\theta(x)$", COLORS["blue"]),
        (0.56, 0.12, 0.22, 0.24, "Perturbed view\n$ f_\\theta(T_s(x))$", COLORS["orange"]),
    ]
    for x, y, w, h, text, color in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
            facecolor="white", edgecolor=color, linewidth=1.4,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    def arrow(p1: tuple[float, float], p2: tuple[float, float]) -> None:
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                                     color="#374151", linewidth=1.0))

    arrow((0.22, 0.715), (0.29, 0.715))
    arrow((0.50, 0.715), (0.57, 0.715))
    arrow((0.76, 0.715), (0.81, 0.715))
    arrow((0.89, 0.59), (0.69, 0.36))
    arrow((0.14, 0.59), (0.33, 0.36))
    ax.text(
        0.50,
        0.06,
        "$\\mathcal{L}=\\frac{1}{2}CE(y,p)+\\frac{1}{2}CE(y,\\tilde p)"
        "+\\lambda D_{SKL}(p,\\tilde p)$",
        ha="center",
        va="center",
        fontsize=9,
        color=COLORS["vermillion"],
    )
    ax.text(0.50, 0.94, "FMC-PC: empirical magnitude-calibrated rotation consistency",
            ha="center", va="center", fontsize=10, weight="bold")
    export_figure(fig, str(out_dir / "figure1_fmc_pc_method"),
                  formats=("pdf", "svg", "png"), size_inches=(7.16, 3.2), dpi=600)
    plt.close(fig)


def figure_main(data_dir: Path, out_dir: Path) -> None:
    setup_style(journal="ieee", lang="en", use_sciplots=False)
    df = pd.read_csv(data_dir / "fmc_pc_seed_results.csv")
    order = ["clean_rcbd", "random_rotation", "pd_pc", "fmc_full_pc", "fmc_angle_pc"]
    labels = ["Clean", "Random\nrotation", "Joint-profile\nPD-PC", "Full-factor\nFMC-PC", "Angle-only\nFMC-PC"]
    colors = [COLORS["grey"], COLORS["blue"], COLORS["orange"], COLORS["purple"], COLORS["green"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), sharey=True)
    _dot_summary(axes[0], df, order, "valid_self_macro_f1", colors, labels)
    _style_axis(axes[0], "Macro-F1 (%)", (0, 100))
    axes[0].set_title("a  Valid-signal Self (subjects 14–17)", loc="left", weight="bold")
    _dot_summary(axes[1], df, order, "as_released_self_macro_f1", colors, labels)
    _style_axis(axes[1], "", (0, 100))
    axes[1].set_title("b  As-released Self (subjects 13–17)", loc="left", weight="bold")
    handles = [plt.Line2D([0], [0], marker=m, color="#374151", markerfacecolor="white",
                          linestyle="none", label=f"seed {s}") for s, m in SEED_MARKERS.items()]
    handles.append(plt.Line2D([0], [0], marker="D", color="#111827", linestyle="none",
                              label="mean ± population SD"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03),
               frameon=False, ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=1.2)
    export_figure(fig, str(out_dir / "figure2_fmc_pc_main_results"),
                  formats=("pdf", "svg", "png"), size_inches=(7.16, 3.0), dpi=600,
                  grayscale_preview=True, tight=False)
    plt.close(fig)


def figure_components(data_dir: Path, out_dir: Path) -> None:
    setup_style(journal="ieee", lang="en", use_sciplots=False)
    df = pd.read_csv(data_dir / "fmc_pc_component_ablation.csv")
    order = ["fmc_angle_pc", "fmc_angle_gain_pc", "fmc_full_no_pc", "fmc_global_pc"]
    labels = ["Angle + PC", "Angle + gain\n+ PC", "Full factors\nwithout PC", "Global angle\n+ PC"]
    colors = [COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["blue"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))
    _dot_summary(axes[0], df, order, "valid_self_macro_f1", colors, labels)
    _style_axis(axes[0], "Macro-F1 (%)", (0, 100))
    axes[0].set_title("a  Valid-signal Self", loc="left", weight="bold")
    _dot_summary(axes[1], df, order, "mutual_subject15_macro_f1", colors, labels)
    _style_axis(axes[1], "Macro-F1 (%)", (0, 100))
    axes[1].set_title("b  Mutual displacement (subject 15)", loc="left", weight="bold")
    fig.tight_layout(w_pad=1.4)
    export_figure(fig, str(out_dir / "figure3_fmc_pc_components"),
                  formats=("pdf", "svg", "png"), size_inches=(7.16, 2.9), dpi=600,
                  grayscale_preview=True, tight=False)
    plt.close(fig)


def figure_backbone(data_dir: Path, out_dir: Path) -> None:
    setup_style(journal="ieee", lang="en", use_sciplots=False)
    df = pd.read_csv(data_dir / "fmc_pc_cross_backbone.csv")
    order = ["clean", "random", "fmc_angle_pc"]
    labels = ["Clean", "Random rotation", "Angle-only FMC-PC"]
    colors = [COLORS["grey"], COLORS["blue"], COLORS["green"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8), sharey=True)
    for ax, backbone, panel in zip(axes, ["RCBD", "DeepConvLSTM"], ["a", "b"], strict=True):
        _dot_summary(ax, df.loc[df["backbone"] == backbone], order,
                     "valid_self_macro_f1", colors, labels)
        _style_axis(ax, "Macro-F1 (%)" if backbone == "RCBD" else "", (0, 100))
        ax.set_title(f"{panel}  {backbone}", loc="left", weight="bold")
    fig.tight_layout(w_pad=1.4)
    export_figure(fig, str(out_dir / "figure4_fmc_pc_cross_backbone"),
                  formats=("pdf", "svg", "png"), size_inches=(7.16, 2.8), dpi=600,
                  grayscale_preview=True, tight=False)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    data_dir = args.package / "data"
    out_dir = args.package / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_method(out_dir)
    figure_main(data_dir, out_dir)
    figure_components(data_dir, out_dir)
    figure_backbone(data_dir, out_dir)


if __name__ == "__main__":
    main()
