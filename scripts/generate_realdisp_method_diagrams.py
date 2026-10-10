from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
GREY = "#666666"
LIGHT = "#F4F6F7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate editable PD-PC method and protocol diagrams."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--scipilot-scripts",
        type=Path,
        default=Path.home() / ".codex" / "skills" / "scipilot-figure-skill" / "scripts",
    )
    return parser.parse_args()


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    *,
    face: str = "white",
    edge: str = GREY,
    fontsize: float = 7.0,
    linewidth: float = 1.0,
    style: str = "round,pad=0.018,rounding_size=0.02",
) -> FancyBboxPatch:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=style,
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#222222",
        linespacing=1.15,
    )
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GREY,
    linewidth: float = 1.1,
    dashed: bool = False,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=linewidth,
            color=color,
            linestyle="--" if dashed else "-",
            connectionstyle=connectionstyle,
            transform=ax.transAxes,
            clip_on=False,
        )
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.03,
        1.02,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )


def export(fig: plt.Figure, basename: Path, size: tuple[float, float]) -> None:
    from export_figure import export_figure
    from visual_qa import audit_layout, render_preview

    preview = basename.parent / "previews" / f"{basename.name}_preview.png"
    preview.parent.mkdir(parents=True, exist_ok=True)
    render_preview(fig, str(preview), dpi=180)
    issues = audit_layout(fig)
    if issues:
        print(f"{basename.name}: layout QA issues={issues}")

    export_figure(
        fig,
        str(basename),
        formats=["pdf", "svg", "png"],
        size_inches=size,
        dpi=600,
        grayscale_preview=True,
        tight=True,
    )
    plt.close(fig)


def method_overview(out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.2), constrained_layout=True)
    for ax in axes:
        ax.set_axis_off()

    ax = axes[0]
    panel_label(ax, "a")
    ax.set_title("Label-free placement profiling", fontsize=9, pad=8)
    box(
        ax,
        (0.02, 0.72),
        (0.40, 0.16),
        "Training subjects\nIdeal acceleration",
        face="#EAF3F8",
        edge=BLUE,
    )
    box(
        ax,
        (0.58, 0.72),
        (0.40, 0.16),
        "Training subjects\nSelf acceleration",
        face="#FFF4DA",
        edge=ORANGE,
    )
    arrow(ax, (0.22, 0.71), (0.38, 0.56), color=BLUE)
    arrow(ax, (0.78, 0.71), (0.62, 0.56), color=ORANGE)
    box(
        ax,
        (0.12, 0.34),
        (0.76, 0.22),
        "Per limb and subject (labels never read)\n"
        "$R$: shortest-arc gravity tilt\n"
        "$g$: clipped RMS ratio;  $\\sigma$: excess first-difference scale",
        face=LIGHT,
        edge=GREY,
        fontsize=6.4,
    )
    arrow(ax, (0.50, 0.33), (0.50, 0.21), color=GREEN)
    box(
        ax,
        (0.19, 0.06),
        (0.62, 0.14),
        "Audited profile bank\n$\\mathcal{B}_\\ell=\\{R,g,\\sigma\\}$",
        face="#E8F5F0",
        edge=GREEN,
        fontsize=7.0,
    )

    ax = axes[1]
    panel_label(ax, "b")
    ax.set_title("Placement-distribution perturbation", fontsize=9, pad=8)
    box(ax, (0.03, 0.73), (0.35, 0.15), "Clean four-limb\nwindow $x$", face="#EAF3F8", edge=BLUE)
    box(
        ax,
        (0.62, 0.73),
        (0.35, 0.15),
        "Profile bank\n$\\mathcal{B}_\\ell$",
        face="#E8F5F0",
        edge=GREEN,
    )
    arrow(ax, (0.20, 0.72), (0.39, 0.57), color=BLUE)
    arrow(ax, (0.80, 0.72), (0.61, 0.57), color=GREEN)
    box(
        ax,
        (0.10, 0.35),
        (0.80, 0.22),
        "For each arm/leg pair, perturb one side\n"
        "with probability $p=0.50$\n"
        "$\\tilde{x}_\\ell=gR x_\\ell+\\epsilon$,  $\\epsilon\\sim\\mathcal{N}(0,\\sigma^2)$",
        face="#FFF4DA",
        edge=ORANGE,
        fontsize=6.5,
    )
    arrow(ax, (0.50, 0.34), (0.50, 0.20), color=ORANGE)
    box(
        ax,
        (0.20, 0.06),
        (0.60, 0.14),
        "Calibrated view $\\tilde{x}$\nLabel $y$ unchanged",
        face="#FFF8E8",
        edge=ORANGE,
    )

    ax = axes[2]
    panel_label(ax, "c")
    ax.set_title("Shared prediction and consistency", fontsize=9, pad=8)
    box(ax, (0.01, 0.75), (0.28, 0.13), "Clean $x$", face="#EAF3F8", edge=BLUE)
    box(ax, (0.01, 0.48), (0.28, 0.13), "Calibrated $\\tilde{x}$", face="#FFF4DA", edge=ORANGE)
    box(
        ax,
        (0.39, 0.59),
        (0.31, 0.20),
        "Shared HAR backbone\n$ f_\\theta$\n(RCBD or DCL)",
        face=LIGHT,
        edge=GREY,
    )
    arrow(ax, (0.29, 0.815), (0.39, 0.72), color=BLUE)
    arrow(ax, (0.29, 0.545), (0.39, 0.65), color=ORANGE)
    box(ax, (0.79, 0.75), (0.20, 0.13), "Logits $z$", face="#EAF3F8", edge=BLUE)
    box(ax, (0.79, 0.48), (0.20, 0.13), "Logits $\\tilde{z}$", face="#FFF4DA", edge=ORANGE)
    arrow(ax, (0.70, 0.72), (0.79, 0.815), color=BLUE)
    arrow(ax, (0.70, 0.65), (0.79, 0.545), color=ORANGE)
    arrow(ax, (0.89, 0.74), (0.76, 0.36), color=VERMILLION)
    arrow(ax, (0.89, 0.47), (0.76, 0.36), color=VERMILLION)
    box(
        ax,
        (0.10, 0.10),
        (0.80, 0.25),
        "$\\mathcal{L}_{\\mathrm{PD-PC}}="
        "0.5\\,\\mathrm{CE}(z,y)+0.5\\,\\mathrm{CE}(\\tilde{z},y)$\n"
        "$+\\;0.10\\,D_{\\mathrm{SKL}}(z,\\tilde{z})+\\mathcal{R}_{\\mathrm{backbone}}$\n"
        "Inference uses the original window only",
        face="#FCEBE7",
        edge=VERMILLION,
        fontsize=6.25,
    )
    export(fig, out / "figure1_pd_pc_method", (7.16, 3.2))


def protocol_diagram(out: Path) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.16, 4.25),
        gridspec_kw={"height_ratios": [1.0, 1.15]},
        constrained_layout=True,
    )
    for ax in axes:
        ax.set_axis_off()

    ax = axes[0]
    panel_label(ax, "a")
    ax.set_title("Frozen participant protocol", fontsize=9, pad=6)
    box(
        ax,
        (0.01, 0.27),
        (0.27, 0.46),
        "Subjects 1–10\n\nIdeal: labeled training\nSelf: unlabeled profile estimation",
        face="#E8F5F0",
        edge=GREEN,
        fontsize=7.2,
    )
    box(
        ax,
        (0.365, 0.27),
        (0.27, 0.46),
        "Subjects 11–12\n\nValidation / early stopping\nPost-freeze sensitivity only",
        face="#FFF4DA",
        edge=ORANGE,
        fontsize=7.2,
    )
    box(
        ax,
        (0.72, 0.27),
        (0.27, 0.46),
        "Subjects 13–17\n\nOne frozen final evaluation\nIdeal and Self scenarios",
        face="#EAF3F8",
        edge=BLUE,
        fontsize=7.2,
    )
    arrow(ax, (0.28, 0.50), (0.365, 0.50), color=GREY)
    arrow(ax, (0.635, 0.50), (0.72, 0.50), color=GREY)
    ax.text(
        0.50,
        0.07,
        "No test-participant information enters model or hyperparameter selection",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.8,
        color=VERMILLION,
    )

    ax = axes[1]
    panel_label(ax, "b")
    ax.set_title("Evaluation and reporting", fontsize=9, pad=6)
    box(
        ax,
        (0.01, 0.42),
        (0.34, 0.32),
        "Ideal / Self evaluation\nControlled vs participant-applied\nsensor placement",
        face="#EEF3F1",
        edge=BLUE,
        fontsize=7.0,
    )
    box(
        ax,
        (0.42, 0.42),
        (0.25, 0.32),
        "Valid-signal main\nSubjects 14–17\nPooled Macro-F1",
        face="#E8F5F0",
        edge=GREEN,
        fontsize=6.9,
    )
    box(
        ax,
        (0.73, 0.42),
        (0.26, 0.32),
        "As-released audit\nSubjects 13–17\nSubject 13 retained",
        face="#FCEBE7",
        edge=VERMILLION,
        fontsize=6.8,
    )
    arrow(ax, (0.35, 0.58), (0.42, 0.58), color=BLUE)
    arrow(ax, (0.67, 0.58), (0.73, 0.58), color=VERMILLION, dashed=True)
    ax.text(
        0.50,
        0.13,
        "Three predefined seeds: 256, 3407, and 114514; raw seed values retained in every method comparison",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.6,
        color=GREY,
    )
    export(fig, out / "figureS4_experimental_protocol", (7.16, 4.25))


def main() -> None:
    args = parse_args()
    if not args.scipilot_scripts.exists():
        raise FileNotFoundError(args.scipilot_scripts)
    sys.path.insert(0, str(args.scipilot_scripts))
    from setup_style import setup_style

    setup_style(journal="ieee", lang="en", use_sciplots=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    method_overview(args.output_dir)
    protocol_diagram(args.output_dir)

    records = []
    for name in ("figure1_pd_pc_method", "figureS4_experimental_protocol"):
        png = args.output_dir / f"{name}.png"
        records.append({"figure": name, "file": str(png)})
    with (args.output_dir / "diagram_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    print(f"Generated {len(records)} editable method/protocol diagrams in {args.output_dir}")


if __name__ == "__main__":
    main()
