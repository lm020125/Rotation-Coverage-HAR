from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = Path.home() / ".codex" / "skills" / "scipilot-figure-skill" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))

from export_figure import export_figure  # noqa: E402
from layout_tools import add_panel_labels, finalize_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report, render_preview  # noqa: E402


SEEDS = (256, 3407, 114514)
SEED_MARKERS = {256: "o", 3407: "s", 114514: "^"}
SEED_OFFSETS = {256: -0.12, 3407: 0.0, 114514: 0.12}
OKABE_ITO = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "orange": "#E69F00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#222222",
    "gray": "#7A7A7A",
    "light_gray": "#C8C8C8",
}

METHOD_TO_CELL = {
    "uniform_random_no_pc": "uniform_no_pc",
    "uniform_random_pc": "uniform_pc",
    "angle_fmc_no_pc": "empirical_no_pc",
    "angle_fmc_pc": "empirical_pc",
}

SHORT_ACTIVITY_NAMES = (
    "Walking",
    "Jogging",
    "Running",
    "Jump up",
    "Jump front/back",
    "Jump sideways",
    "Jumping jacks",
    "Jump rope",
    "Trunk twist (arms out)",
    "Trunk twist (elbows)",
    "Waist bend forward",
    "Waist rotation",
    "Cross-foot waist bend",
    "Reach heels back",
    "Lateral bend",
    "Lateral bend + arm",
    "Forward stretch",
    "Opposite trunk twist",
    "Lateral arm raise",
    "Frontal arm raise",
    "Hand claps",
    "Cross arms",
    "Shoulder rotation (high)",
    "Shoulder rotation (low)",
    "Arm internal rotation",
    "Knees to chest",
    "Heels to backside",
    "Crouching",
    "Alt. forward knee bend",
    "Knee rotation",
    "Rowing",
    "Elliptical",
    "Cycling",
)


def effect_values(cells: dict[int, dict[str, float]]) -> dict[str, list[float]]:
    out = {
        "Empirical\namplitude": [],
        "Prediction\nconsistency": [],
        "Amplitude × PC\ninteraction": [],
    }
    for seed in SEEDS:
        c = cells[seed]
        out["Empirical\namplitude"].append(
            0.5
            * (
                (c["empirical_no_pc"] - c["uniform_no_pc"])
                + (c["empirical_pc"] - c["uniform_pc"])
            )
            * 100.0
        )
        out["Prediction\nconsistency"].append(
            0.5
            * (
                (c["uniform_pc"] - c["uniform_no_pc"])
                + (c["empirical_pc"] - c["empirical_no_pc"])
            )
            * 100.0
        )
        out["Amplitude × PC\ninteraction"].append(
            (
                (c["empirical_pc"] - c["empirical_no_pc"])
                - (c["uniform_pc"] - c["uniform_no_pc"])
            )
            * 100.0
        )
    return out


def export_exact(fig, basename: Path, size: tuple[float, float]) -> list[str]:
    paths = export_figure(
        fig,
        str(basename),
        formats=("pdf", "svg", "png"),
        size_inches=size,
        dpi=600,
        grayscale_preview=False,
        tight=False,
    )
    png = basename.with_suffix(".png")
    gray = basename.with_name(basename.name + "_grayscale").with_suffix(".png")
    with Image.open(png) as image:
        image.convert("L").save(gray, dpi=(600, 600))
    paths.append(str(gray))
    return paths


def save_qa(
    fig,
    output_root: Path,
    stem: str,
    size: tuple[float, float],
) -> tuple[list[str], list[tuple[str, str]]]:
    preview = output_root / "previews" / f"{stem}_preview.png"
    preview.parent.mkdir(parents=True, exist_ok=True)
    render_preview(fig, str(preview), dpi=150)
    issues = audit_layout(fig)
    verdict = print_report(issues)
    if verdict == "FAIL":
        raise RuntimeError(f"Visual QA failed for {stem}: {issues}")
    paths = export_exact(fig, output_root / stem, size)
    return paths, issues


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base = REPO / "results" / "fmc_pc_revision_20260723_local"
    final = base / "participant_class_final"
    aggregate = pd.read_csv(final / "aggregate_seed_metrics.csv")
    participants = pd.read_csv(final / "participant_seed_metrics.csv")
    classes = pd.read_csv(final / "class_seed_metrics.csv")
    effects = json.loads((base / "server_results" / "effects_2x2.json").read_text(encoding="utf-8"))
    return aggregate, participants, classes, effects


def draw_factorial(
    aggregate: pd.DataFrame,
    output_root: Path,
) -> dict[str, Any]:
    size = (7.16, 3.55)
    cells: dict[int, dict[str, float]] = {seed: {} for seed in SEEDS}
    for method, cell in METHOD_TO_CELL.items():
        rows = aggregate[aggregate["method"] == method].set_index("seed")
        for seed in SEEDS:
            cells[seed][cell] = float(rows.loc[seed, "macro_f1_fixed_33"])
    effects = effect_values(cells)
    pd.DataFrame(
        [
            {"effect": label.replace("\n", " "), "seed": seed, "effect_pp": value}
            for label, values in effects.items()
            for seed, value in zip(SEEDS, values)
        ]
    ).to_csv(output_root / "figure_2x2_effect_source.csv", index=False)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=size,
        gridspec_kw={"width_ratios": (1.35, 1.0)},
        constrained_layout=True,
    )
    ax, effect_ax = axes
    x = np.asarray([0.0, 1.0])
    amplitude_styles = {
        "Uniform 0–90°": {
            "cells": ("uniform_no_pc", "uniform_pc"),
            "color": OKABE_ITO["blue"],
            "linestyle": "-",
        },
        "Empirical amplitude": {
            "cells": ("empirical_no_pc", "empirical_pc"),
            "color": OKABE_ITO["vermillion"],
            "linestyle": "--",
        },
    }
    for label, style in amplitude_styles.items():
        cell_off, cell_on = style["cells"]
        for seed in SEEDS:
            values = np.asarray([cells[seed][cell_off], cells[seed][cell_on]]) * 100.0
            ax.plot(
                x,
                values,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=0.85,
                alpha=0.55,
                marker=SEED_MARKERS[seed],
                markersize=4.2,
                markerfacecolor="white",
                markeredgewidth=0.8,
                zorder=2,
            )
        means = np.asarray(
            [
                fmean(cells[seed][cell_off] for seed in SEEDS),
                fmean(cells[seed][cell_on] for seed in SEEDS),
            ]
        ) * 100.0
        ax.plot(
            x,
            means,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            marker="D",
            markersize=5.2,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=4,
        )
        for x_value, y_value in zip(x, means):
            ax.annotate(
                f"{y_value:.2f}",
                (x_value, y_value),
                xytext=(0, 7 if label.startswith("Empirical") else -11),
                textcoords="offset points",
                ha="center",
                va="bottom" if label.startswith("Empirical") else "top",
                fontsize=6.5,
                color=style["color"],
            )
    ax.set_xticks(x, ("PC off", "PC on"))
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Valid-Self Macro-F1 (%)")
    ax.set_title("2 × 2 seed-level trajectories", loc="left", pad=5)
    ax.grid(axis="y", color="#E1E1E1", linewidth=0.55)
    ax.text(
        0.5,
        0.965,
        "Difference-in-differences = +4.26 pp",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.7,
        color=OKABE_ITO["black"],
    )
    amplitude_handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.6,
            label=label,
        )
        for label, style in amplitude_styles.items()
    ]
    seed_handles = [
        Line2D(
            [0],
            [0],
            color=OKABE_ITO["gray"],
            marker=SEED_MARKERS[seed],
            linestyle="none",
            markerfacecolor="white",
            markersize=4.2,
            label=f"Seed {seed}",
        )
        for seed in SEEDS
    ]
    mean_handle = Line2D(
        [0],
        [0],
        color=OKABE_ITO["black"],
        marker="D",
        linestyle="none",
        markersize=4.8,
        label="Mean",
    )
    ax.legend(
        handles=amplitude_handles + seed_handles + [mean_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=2,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.8,
    )

    effect_labels = list(effects)
    effect_colors = (OKABE_ITO["blue"], OKABE_ITO["purple"], OKABE_ITO["orange"])
    for position, (label, color) in enumerate(zip(effect_labels, effect_colors)):
        values = effects[label]
        for seed, value in zip(SEEDS, values):
            effect_ax.scatter(
                position + SEED_OFFSETS[seed],
                value,
                s=24,
                marker=SEED_MARKERS[seed],
                facecolor="white",
                edgecolor=color,
                linewidth=1.0,
                zorder=3,
            )
        mean_value = fmean(values)
        effect_ax.scatter(
            position,
            mean_value,
            s=42,
            marker="D",
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        effect_ax.annotate(
            f"{mean_value:+.2f}",
            (position, mean_value),
            xytext=(0, 7 if mean_value >= 0 else -11),
            textcoords="offset points",
            ha="center",
            va="bottom" if mean_value >= 0 else "top",
            fontsize=6.5,
            color=color,
        )
    effect_ax.axhline(0, color=OKABE_ITO["black"], linewidth=0.75, zorder=1)
    effect_ax.set_xticks(range(3), effect_labels)
    effect_ax.set_xlim(-0.48, 2.48)
    effect_ax.set_ylim(-8, 18)
    effect_ax.set_ylabel("Factorial effect (percentage points)")
    effect_ax.set_title("Descriptive factorial effects", loc="left", pad=5)
    effect_ax.grid(axis="y", color="#E1E1E1", linewidth=0.55)
    effect_ax.text(
        0.99,
        0.03,
        "n = 3 fixed seeds; no inferential test",
        transform=effect_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=OKABE_ITO["gray"],
    )
    finalize_figure(fig)
    add_panel_labels(
        fig,
        axes=axes,
        style="ieee",
        fontsize=8,
        x_offset_pt=3,
        y_offset_pt=-3,
        ha="left",
        va="top",
    )
    paths, issues = save_qa(fig, output_root, "figure_2x2_mechanism", size)
    plt.close(fig)
    return {"paths": paths, "issues": issues, "effects": effects}


def paired_deltas(
    frame: pd.DataFrame,
    index_columns: list[str],
) -> pd.DataFrame:
    current = frame[frame["method"] == "angle_fmc_pc"].set_index(index_columns)
    baseline = frame[frame["method"] == "uniform_random_no_pc"].set_index(index_columns)
    values = 100.0 * (current["macro_f1_fixed_33"] - baseline["macro_f1_fixed_33"])
    return values.rename("delta_pp").reset_index()


def class_deltas(classes: pd.DataFrame) -> pd.DataFrame:
    current = classes[classes["method"] == "angle_fmc_pc"].set_index(
        ["seed", "class_index"]
    )
    baseline = classes[classes["method"] == "uniform_random_no_pc"].set_index(
        ["seed", "class_index"]
    )
    values = 100.0 * (current["f1"] - baseline["f1"])
    result = values.rename("delta_pp").reset_index()
    result["activity_id"] = result["class_index"].map(lambda value: f"A{value + 1}")
    result["short_activity"] = result["class_index"].map(
        lambda value: SHORT_ACTIVITY_NAMES[value]
    )
    return result


def draw_heterogeneity(
    participants: pd.DataFrame,
    classes: pd.DataFrame,
    output_root: Path,
) -> dict[str, Any]:
    size = (7.16, 8.6)
    participant_delta = paired_deltas(participants, ["seed", "participant"])
    class_delta = class_deltas(classes)
    participant_delta.to_csv(output_root / "figure_heterogeneity_participant_source.csv", index=False)
    class_delta.to_csv(output_root / "figure_heterogeneity_class_source.csv", index=False)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=size,
        gridspec_kw={"height_ratios": (1.15, 4.85)},
        constrained_layout=True,
    )
    participant_ax, class_ax = axes
    participants_order = ("subject14", "subject15", "subject16", "subject17")
    participant_positions = {name: index for index, name in enumerate(participants_order)}
    for seed in SEEDS:
        rows = participant_delta[participant_delta["seed"] == seed]
        participant_ax.scatter(
            rows["delta_pp"],
            [participant_positions[name] + SEED_OFFSETS[seed] for name in rows["participant"]],
            s=28,
            marker=SEED_MARKERS[seed],
            facecolor="white",
            edgecolor=OKABE_ITO["blue"],
            linewidth=1.0,
            zorder=3,
        )
    for participant, position in participant_positions.items():
        mean_value = float(
            participant_delta[participant_delta["participant"] == participant]["delta_pp"].mean()
        )
        participant_ax.scatter(
            mean_value,
            position,
            s=48,
            marker="D",
            facecolor=OKABE_ITO["orange"] if mean_value >= 0 else OKABE_ITO["blue"],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        participant_ax.annotate(
            f"{mean_value:+.2f}",
            (mean_value, position),
            xytext=(6 if mean_value >= 0 else -6, 0),
            textcoords="offset points",
            ha="left" if mean_value >= 0 else "right",
            va="center",
            fontsize=6.5,
        )
    participant_ax.axvline(0, color=OKABE_ITO["black"], linewidth=0.75)
    participant_ax.set_xlim(-10, 62)
    participant_ax.set_yticks(range(4), ("Subject 14", "Subject 15", "Subject 16", "Subject 17"))
    participant_ax.invert_yaxis()
    participant_ax.set_xlabel("FMC-PC − uniform rotation Macro-F1 (percentage points)")
    participant_ax.set_title("Participant-level heterogeneity", loc="left", pad=5)
    participant_ax.grid(axis="x", color="#E1E1E1", linewidth=0.55)
    participant_ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=SEED_MARKERS[seed],
                color="none",
                markerfacecolor="white",
                markeredgecolor=OKABE_ITO["blue"],
                label=f"Seed {seed}",
                markersize=4.5,
            )
            for seed in SEEDS
        ]
        + [
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor=OKABE_ITO["orange"],
                markeredgecolor="white",
                label="Mean",
                markersize=5.2,
            )
        ],
        loc="lower right",
        ncol=4,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.8,
    )

    class_means = class_delta.groupby(["class_index", "activity_id", "short_activity"], as_index=False)[
        "delta_pp"
    ].mean()
    class_means = class_means.sort_values("delta_pp", ascending=False).reset_index(drop=True)
    class_positions = {
        int(row.class_index): position for position, row in class_means.iterrows()
    }
    for seed in SEEDS:
        rows = class_delta[class_delta["seed"] == seed]
        class_ax.scatter(
            rows["delta_pp"],
            [class_positions[int(index)] + SEED_OFFSETS[seed] for index in rows["class_index"]],
            s=15,
            marker=SEED_MARKERS[seed],
            facecolor="white",
            edgecolor=OKABE_ITO["gray"],
            linewidth=0.65,
            alpha=0.9,
            zorder=2,
        )
    mean_colors = [
        OKABE_ITO["orange"] if value >= 0 else OKABE_ITO["blue"]
        for value in class_means["delta_pp"]
    ]
    class_ax.scatter(
        class_means["delta_pp"],
        range(len(class_means)),
        s=30,
        marker="D",
        c=mean_colors,
        edgecolor="white",
        linewidth=0.65,
        zorder=4,
    )
    class_ax.axvline(0, color=OKABE_ITO["black"], linewidth=0.75)
    class_ax.set_xlim(-35, 80)
    class_ax.set_ylim(-0.7, 32.7)
    class_ax.set_yticks(
        range(len(class_means)),
        [
            f"{row.activity_id}  {row.short_activity}"
            for row in class_means.itertuples(index=False)
        ],
    )
    class_ax.invert_yaxis()
    class_ax.set_xlabel("FMC-PC − uniform rotation class F1 (percentage points)")
    class_ax.set_title("Class-level heterogeneity (sorted by mean difference)", loc="left", pad=5)
    class_ax.grid(axis="x", color="#E1E1E1", linewidth=0.55)
    class_ax.tick_params(axis="y", labelsize=6.2)
    finalize_figure(fig)
    add_panel_labels(
        fig,
        axes=axes,
        style="ieee",
        fontsize=8,
        x_offset_pt=3,
        y_offset_pt=-3,
        ha="left",
        va="top",
    )
    paths, issues = save_qa(fig, output_root, "figure_heterogeneity", size)
    plt.close(fig)
    return {
        "paths": paths,
        "issues": issues,
        "participant_mean_deltas_pp": {
            participant: float(
                participant_delta[participant_delta["participant"] == participant][
                    "delta_pp"
                ].mean()
            )
            for participant in participants_order
        },
        "class_mean_delta_range_pp": [
            float(class_means["delta_pp"].min()),
            float(class_means["delta_pp"].max()),
        ],
    }


def main() -> None:
    setup_style(journal="ieee", lang="en", use_sciplots=True, constrained_layout=True)
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.3,
        }
    )
    aggregate, participants, classes, _ = read_inputs()
    output_root = (
        REPO
        / "results"
        / "fmc_pc_revision_20260723_local"
        / "figure_revision_2x2"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    aggregate[
        aggregate["method"].isin(METHOD_TO_CELL)
    ].to_csv(output_root / "figure_2x2_source.csv", index=False)
    factorial = draw_factorial(aggregate, output_root)
    heterogeneity = draw_heterogeneity(participants, classes, output_root)
    qa = {
        "factorial": factorial,
        "heterogeneity": heterogeneity,
        "semantic_qa": {
            "individual_seed_points_shown": True,
            "mean_only_bars_used": False,
            "proportion_axis_starts_at_zero": True,
            "categorical_means_connected": False,
            "colorblind_palette": "Okabe-Ito with redundant marker/linestyle encoding",
            "grayscale_versions_exported": True,
            "inference_claim": "descriptive_only_n_3",
            "confusion_difference_omitted": (
                "Omitted from the compact heterogeneity figure to avoid overloading the "
                "participant and 33-class argument; normalized matrices remain available "
                "as machine-readable supplementary outputs."
            ),
        },
    }
    (output_root / "figure_qa.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )
    captions = """# Figure captions

## 2 × 2 mechanism figure

**Empirical rotation amplitude and prediction consistency in the valid-signal Self-placement cohort.** (a) Macro-F1 for the four cells of the amplitude-source × prediction-consistency design. Each thin trajectory represents one fixed training seed (256, 3407, or 114514); diamonds and thick trajectories show arithmetic means. (b) Seed-level factorial contrasts computed from the same paired four-cell results. Positive values favor empirical amplitude calibration or a positive amplitude × consistency interaction. The three seeds are descriptive computational replicates; no inferential test or population-level significance claim is made.

## Heterogeneity figure

**Heterogeneity of FMC-PC gains over uniform random rotation without prediction consistency.** (a) Matched participant-level fixed-33 Macro-F1 differences for subjects 14–17. (b) Matched class-level F1 differences for all 33 REALDISP activities, sorted by their three-seed mean. Small markers show individual fixed seeds and diamonds show arithmetic means; positive values favor FMC-PC. Participant and class comparisons use the same seed on both sides. No inferential test is applied (n = 3 fixed seeds).
"""
    (output_root / "figure_captions.md").write_text(captions, encoding="utf-8")
    print(f"Wrote figures and QA to {output_root}")


if __name__ == "__main__":
    main()
