from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


OKABE = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#7F7F7F",
    "light_grey": "#B8B8B8",
}

ACTIVITIES = [
    "Walking",
    "Jogging",
    "Running",
    "Jump up",
    "Jump front & back",
    "Jump sideways",
    "Jump legs/arms open/closed",
    "Jump rope",
    "Trunk twist (arms outstretched)",
    "Trunk twist (elbows bent)",
    "Waist bends forward",
    "Waist rotation",
    "Waist bends (opposite hand to foot)",
    "Reach heels backwards",
    "Lateral bend",
    "Lateral bend with arm up",
    "Repetitive forward stretching",
    "Upper/lower body opposite twist",
    "Lateral elevation of arms",
    "Frontal elevation of arms",
    "Frontal hand claps",
    "Frontal crossing of arms",
    "Shoulders high-amplitude rotation",
    "Shoulders low-amplitude rotation",
    "Arms inner rotation",
    "Alternating knees to breast",
    "Alternating heels to backside",
    "Knees bending (crouching)",
    "Alternating knees bending forward",
    "Rotation on the knees",
    "Rowing",
    "Elliptical bike",
    "Cycling",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-grade REALDISP manuscript figures."
    )
    parser.add_argument(
        "archive_root",
        type=Path,
        help="Path to the archived REALDISP four-limb protocol directory.",
    )
    parser.add_argument(
        "--scipilot-scripts",
        type=Path,
        default=Path.home() / ".codex" / "skills" / "scipilot-figure-skill" / "scripts",
    )
    return parser.parse_args()


def add_scipilot_to_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"SciPilot scripts not found: {path}")
    sys.path.insert(0, str(path))


def seed_offsets(seeds: list[int], width: float = 0.13) -> dict[int, float]:
    if len(seeds) == 1:
        return {seeds[0]: 0.0}
    values = np.linspace(-width, width, len(seeds))
    return dict(zip(sorted(seeds), values, strict=True))


def point_interval_horizontal(
    ax: plt.Axes,
    data: pd.DataFrame,
    value: str,
    methods: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
    markers: dict[str, str],
) -> None:
    seeds = sorted(data["seed"].astype(int).unique().tolist())
    offsets = seed_offsets(seeds, 0.14)
    for y, method in enumerate(methods):
        sub = data[data["method"] == method].copy()
        values = sub[value].astype(float).to_numpy()
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=0))
        for row in sub.itertuples(index=False):
            ax.scatter(
                float(getattr(row, value)),
                y + offsets[int(row.seed)],
                s=24,
                marker=markers[method],
                facecolor=colors[method] if method != "clean_rcbd" else "white",
                edgecolor=colors[method],
                linewidth=0.75,
                alpha=0.9,
                zorder=3,
            )
        ax.errorbar(
            mean,
            y,
            xerr=sd,
            fmt=markers[method],
            markersize=6.0,
            color=colors[method],
            markerfacecolor=colors[method] if method != "clean_rcbd" else "white",
            markeredgewidth=0.9,
            elinewidth=1.1,
            capsize=2.4,
            zorder=4,
        )
        label_x = min(max(float(values.max()), mean + sd) + 1.4, 96.5)
        ax.text(
            label_x,
            y,
            f"{mean:.2f}",
            va="center",
            ha="left",
            fontsize=6.6,
            color=colors[method],
        )
    ax.set_yticks(range(len(methods)), [labels[m] for m in methods])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Macro-F1 (%)")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.55, zorder=0)


def export_with_qa(
    fig: plt.Figure,
    basename: Path,
    size: tuple[float, float],
    qa_records: list[dict[str, object]],
    panel_labels: bool = False,
) -> None:
    from export_figure import export_figure
    from layout_tools import add_panel_labels, finalize_figure
    from visual_qa import audit_layout, render_preview

    finalize_figure(fig)
    if panel_labels:
        add_panel_labels(fig, style="ieee")
    preview = basename.parent / "previews" / f"{basename.name}_preview.png"
    render_preview(fig, str(preview), dpi=180)
    issues = audit_layout(fig)
    qa_records.append(
        {
            "figure": basename.name,
            "preview": str(preview),
            "issues": [{"severity": level, "message": msg} for level, msg in issues],
        }
    )
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


def figure_2(ablation: pd.DataFrame, out: Path, qa: list[dict[str, object]]) -> None:
    methods = [
        "clean_rcbd",
        "random_rotation",
        "pd_rotation_only",
        "pd_rotation_gain",
        "pd_full_no_pc",
        "random_rotation_pc",
        "pd_pc",
    ]
    labels = {
        "clean_rcbd": "Clean RCBD",
        "random_rotation": "Random rotation",
        "pd_rotation_only": "PD rotation",
        "pd_rotation_gain": "PD rotation + gain",
        "pd_full_no_pc": "PD full (no PC)",
        "random_rotation_pc": "Random rotation + PC",
        "pd_pc": "PD-PC",
    }
    colors = {
        "clean_rcbd": OKABE["black"],
        "random_rotation": OKABE["orange"],
        "pd_rotation_only": "#999999",
        "pd_rotation_gain": "#777777",
        "pd_full_no_pc": "#555555",
        "random_rotation_pc": OKABE["purple"],
        "pd_pc": OKABE["blue"],
    }
    markers = {
        "clean_rcbd": "o",
        "random_rotation": "^",
        "pd_rotation_only": "v",
        "pd_rotation_gain": "D",
        "pd_full_no_pc": "P",
        "random_rotation_pc": "X",
        "pd_pc": "s",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.55), constrained_layout=True)
    point_interval_horizontal(
        axes[0],
        ablation,
        "final_ideal_macro_f1",
        methods,
        labels,
        colors,
        markers,
    )
    point_interval_horizontal(
        axes[1],
        ablation,
        "final_self_macro_f1",
        methods,
        labels,
        colors,
        markers,
    )
    axes[0].set_title("Ideal placement")
    axes[1].set_title("Self-placement")
    axes[1].set_yticklabels([])
    axes[1].set_ylabel("")
    export_with_qa(fig, out / "figure2_ablation_tradeoff", (7.16, 3.55), qa, True)


def figure_3(participants: pd.DataFrame, out: Path, qa: list[dict[str, object]]) -> None:
    methods = ["clean_rcbd", "random_rotation", "pd_pc"]
    labels = ["Clean", "Random\nrotation", "PD-PC"]
    colors = {
        "clean_rcbd": OKABE["black"],
        "random_rotation": OKABE["orange"],
        "pd_pc": OKABE["blue"],
    }
    markers = {"clean_rcbd": "o", "random_rotation": "^", "pd_pc": "s"}
    subjects = [f"subject{i}" for i in range(13, 18)]
    seeds = sorted(participants["seed"].astype(int).unique().tolist())
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 5.0), sharey=True, constrained_layout=True)
    for ax, subject in zip(axes.flat, subjects, strict=False):
        sub = participants[participants["participant"] == subject]
        for seed in seeds:
            row = sub[sub["seed"] == seed].set_index("method")
            y = [float(row.loc[m, "macro_f1"]) for m in methods]
            ax.plot(
                range(3),
                y,
                color="#B0B0B0",
                linewidth=0.7,
                alpha=0.7,
                zorder=1,
            )
            for x, method in enumerate(methods):
                ax.scatter(
                    x,
                    y[x],
                    s=25,
                    marker=markers[method],
                    facecolor=colors[method] if method != "clean_rcbd" else "white",
                    edgecolor=colors[method],
                    linewidth=0.8,
                    zorder=2,
                )
        means = sub.groupby("method")["macro_f1"].mean()
        for x, method in enumerate(methods):
            ax.scatter(
                x,
                float(means[method]),
                s=62,
                marker=markers[method],
                facecolor=colors[method] if method != "clean_rcbd" else "white",
                edgecolor=colors[method],
                linewidth=1.2,
                zorder=3,
            )
        ax.set_xticks(range(3), labels)
        ax.set_ylim(0, 100)
        ax.set_title(subject.replace("subject", "Subject "))
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
        if subject == "subject13":
            ax.text(
                0.5,
                0.12,
                "All-zero four-limb Self record\n(as-released audit only)",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=OKABE["vermillion"],
            )
    # Fill the sixth panel with the valid-signal participant mean for each seed.
    # Subject 13 remains visible as an input-quality audit but is excluded from
    # the main subjects 14--17 aggregate shown here.
    ax = axes.flat[-1]
    valid_participants = participants[participants["participant"] != "subject13"]
    seed_means = valid_participants.groupby(["seed", "method"], as_index=False)["macro_f1"].mean()
    for seed in seeds:
        row = seed_means[seed_means["seed"] == seed].set_index("method")
        y = [float(row.loc[m, "macro_f1"]) for m in methods]
        ax.plot(range(3), y, color="#B0B0B0", linewidth=0.7, alpha=0.7, zorder=1)
        for x, method in enumerate(methods):
            ax.scatter(
                x,
                y[x],
                s=25,
                marker=markers[method],
                facecolor=colors[method] if method != "clean_rcbd" else "white",
                edgecolor=colors[method],
                linewidth=0.8,
                zorder=2,
            )
    means = seed_means.groupby("method")["macro_f1"].mean()
    for x, method in enumerate(methods):
        ax.scatter(
            x,
            float(means[method]),
            s=62,
            marker=markers[method],
            facecolor=colors[method] if method != "clean_rcbd" else "white",
            edgecolor=colors[method],
            linewidth=1.2,
            zorder=3,
        )
    ax.set_xticks(range(3), labels)
    ax.set_ylim(0, 100)
    ax.set_title("Valid-signal mean")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    axes[0, 0].set_ylabel("Self Macro-F1 (%)")
    axes[1, 0].set_ylabel("Self Macro-F1 (%)")
    export_with_qa(fig, out / "figure3_participant_heterogeneity", (7.16, 5.0), qa, True)


def plot_sensitivity_axis(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    x_values: list[float],
    y_col: str,
    x_label: str,
) -> None:
    sub = data[data[x_col].isin(x_values)].copy()
    other = (
        "consistency_weight" if x_col == "augmentation_probability" else "augmentation_probability"
    )
    fixed = 0.10 if x_col == "augmentation_probability" else 0.50
    sub = sub[np.isclose(sub[other].astype(float), fixed)]
    seeds = sorted(sub["seed"].astype(int).unique().tolist())
    offsets = seed_offsets(seeds, 0.012 if x_col == "augmentation_probability" else 0.004)
    means, sds = [], []
    for x in x_values:
        values = sub[np.isclose(sub[x_col].astype(float), x)][y_col].astype(float)
        means.append(float(values.mean()))
        sds.append(float(values.std(ddof=0)))
        for seed, value in zip(
            sub[np.isclose(sub[x_col].astype(float), x)]["seed"].astype(int),
            values,
            strict=True,
        ):
            ax.scatter(
                x + offsets[int(seed)],
                float(value),
                s=25,
                marker="o",
                facecolor="white",
                edgecolor=OKABE["grey"],
                linewidth=0.8,
                zorder=3,
            )
    ax.plot(
        x_values,
        means,
        color=OKABE["blue"],
        marker="s",
        linewidth=1.15,
        markersize=5.5,
        zorder=4,
    )
    ax.errorbar(
        x_values,
        means,
        yerr=sds,
        fmt="none",
        color=OKABE["blue"],
        elinewidth=1.0,
        capsize=2.5,
        zorder=3,
    )
    center_x = 0.50 if x_col == "augmentation_probability" else 0.10
    center_y = means[x_values.index(center_x)]
    ax.scatter(
        [center_x],
        [center_y],
        s=92,
        marker="*",
        color=OKABE["blue"],
        edgecolor="black",
        linewidth=0.55,
        zorder=5,
    )
    annotation_offset = -12 if center_y > 88 else 8
    ax.annotate(
        "Frozen center",
        (center_x, center_y),
        xytext=(4, annotation_offset),
        textcoords="offset points",
        fontsize=6.5,
        color=OKABE["blue"],
        va="top" if annotation_offset < 0 else "bottom",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x_values)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)


def figure_4(sensitivity: pd.DataFrame, out: Path, qa: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.35), constrained_layout=True)
    plot_sensitivity_axis(
        axes[0],
        sensitivity,
        "augmentation_probability",
        [0.25, 0.50, 0.75],
        "validation_self_macro_f1",
        "Augmentation probability, p",
    )
    plot_sensitivity_axis(
        axes[1],
        sensitivity,
        "consistency_weight",
        [0.05, 0.10, 0.20],
        "validation_self_macro_f1",
        "Consistency weight, lambda",
    )
    axes[1].set_ylabel("")
    axes[0].set_title("Varying p (lambda = 0.10)")
    axes[1].set_title("Varying lambda (p = 0.50)")
    fig.suptitle("Post-freeze validation-only sensitivity", fontsize=9)
    export_with_qa(fig, out / "figure4_hyperparameter_sensitivity", (7.16, 3.35), qa, True)


def figure_5(backbone: pd.DataFrame, out: Path, qa: list[dict[str, object]]) -> None:
    conditions = ["Clean", "Random rotation", "PD-PC"]
    colors = [OKABE["black"], OKABE["orange"], OKABE["blue"]]
    markers = ["o", "^", "s"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.35), sharey=True, constrained_layout=True)
    for ax, model in zip(axes, ["RCBD", "DeepConvLSTM"], strict=True):
        sub = backbone[backbone["model"] == model]
        for x, (condition, color, marker) in enumerate(
            zip(conditions, colors, markers, strict=True)
        ):
            values = sub[sub["condition"] == condition]["final_self_macro_f1"].astype(float)
            offsets = np.linspace(-0.10, 0.10, len(values))
            for dx, value in zip(offsets, values, strict=True):
                ax.scatter(
                    x + dx,
                    float(value),
                    s=26,
                    marker=marker,
                    facecolor=color if condition != "Clean" else "white",
                    edgecolor=color,
                    linewidth=0.8,
                    zorder=3,
                )
            ax.errorbar(
                x,
                float(values.mean()),
                yerr=float(values.std(ddof=0)),
                fmt=marker,
                markersize=7,
                color=color,
                markerfacecolor=color if condition != "Clean" else "white",
                markeredgewidth=1.0,
                elinewidth=1.1,
                capsize=2.5,
                zorder=4,
            )
        ax.set_xticks(range(3), ["Clean", "Random\nrotation", "PD-PC"])
        ax.set_ylim(0, 100)
        ax.set_title(model)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    axes[0].set_ylabel("Self Macro-F1 (%)")
    export_with_qa(fig, out / "figure5_cross_backbone", (7.16, 3.35), qa, True)


def figure_s1(classes: pd.DataFrame, out: Path, qa: list[dict[str, object]]) -> None:
    pivot = classes.pivot(index="activity", columns="method", values="mean_f1")
    order = [f"A{i}" for i in range(1, 34)]
    pivot = pivot.reindex(order)
    matrix = pd.DataFrame(
        {
            "PD-PC - Clean": pivot["pd_pc"] - pivot["clean_rcbd"],
            "PD-PC - Random rotation": pivot["pd_pc"] - pivot["random_rotation"],
        },
        index=order,
    )
    labels = [f"A{i}: {name}" for i, name in enumerate(ACTIVITIES, start=1)]
    vmax = float(np.ceil(np.abs(matrix.to_numpy()).max() / 5.0) * 5.0)
    fig, ax = plt.subplots(figsize=(7.16, 9.4), constrained_layout=True)
    hm = sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.25,
        linecolor="white",
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 6},
        yticklabels=labels,
        cbar_kws={"label": "Macro-F1 difference (pp)", "shrink": 0.72},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=6.1)
    ax.tick_params(axis="x", labelsize=7, rotation=0)
    hm.collections[0].colorbar.ax.tick_params(labelsize=7)
    export_with_qa(fig, out / "figureS1_classwise_differences", (7.16, 9.4), qa)


def figure_s2(sensitivity: pd.DataFrame, out: Path, qa: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.55), constrained_layout=True)
    rows = [
        ("validation_ideal_macro_f1", "Ideal placement"),
        ("validation_rotation_stress_macro_f1", "Rotation stress"),
    ]
    for row_idx, (metric, title) in enumerate(rows):
        plot_sensitivity_axis(
            axes[row_idx, 0],
            sensitivity,
            "augmentation_probability",
            [0.25, 0.50, 0.75],
            metric,
            "Augmentation probability, p",
        )
        plot_sensitivity_axis(
            axes[row_idx, 1],
            sensitivity,
            "consistency_weight",
            [0.05, 0.10, 0.20],
            metric,
            "Consistency weight, lambda",
        )
        axes[row_idx, 0].set_title(f"{title}: varying p")
        axes[row_idx, 1].set_title(f"{title}: varying lambda")
        axes[row_idx, 1].set_ylabel("")
    export_with_qa(fig, out / "figureS2_sensitivity_other_scenarios", (7.16, 5.55), qa, True)


def figure_s3(ablation: pd.DataFrame, out: Path, qa: list[dict[str, object]]) -> None:
    methods = [
        "clean_rcbd",
        "random_rotation",
        "pd_rotation_only",
        "pd_rotation_gain",
        "pd_full_no_pc",
        "random_rotation_pc",
        "pd_pc",
    ]
    labels = {
        "clean_rcbd": "Clean RCBD",
        "random_rotation": "Random rotation",
        "pd_rotation_only": "PD rotation",
        "pd_rotation_gain": "PD rotation + gain",
        "pd_full_no_pc": "PD full (no PC)",
        "random_rotation_pc": "Random rotation + PC",
        "pd_pc": "PD-PC",
    }
    colors = {m: "#777777" for m in methods}
    colors.update(
        {
            "clean_rcbd": OKABE["black"],
            "random_rotation": OKABE["orange"],
            "pd_pc": OKABE["blue"],
        }
    )
    markers = dict(zip(methods, ["o", "^", "v", "D", "P", "X", "s"], strict=True))
    fig, ax = plt.subplots(figsize=(7.16, 3.85), constrained_layout=True)
    point_interval_horizontal(
        ax,
        ablation,
        "final_self_accuracy",
        methods,
        labels,
        colors,
        markers,
    )
    ax.set_xlabel("Self Accuracy (%)")
    export_with_qa(fig, out / "figureS3_self_accuracy", (7.16, 3.85), qa)


def main() -> int:
    args = parse_args()
    add_scipilot_to_path(args.scipilot_scripts)
    from setup_style import setup_style

    setup_style(journal="ieee", lang="en", use_sciplots=False)
    plt.rcParams.update(
        {
            "figure.constrained_layout.use": True,
            "axes.axisbelow": True,
            "legend.frameon": False,
        }
    )
    data_root = args.archive_root / "paper_assets" / "data"
    out = args.archive_root / "paper_assets" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    (out / "previews").mkdir(parents=True, exist_ok=True)

    ablation = pd.read_csv(data_root / "ablation_runs.csv")
    participants = pd.read_csv(data_root / "participant_self_runs.csv")
    sensitivity = pd.read_csv(data_root / "sensitivity_runs.csv")
    backbone = pd.read_csv(data_root / "backbone_runs.csv")
    classes = pd.read_csv(data_root / "class_self_main.csv")

    qa: list[dict[str, object]] = []
    figure_2(ablation, out, qa)
    figure_3(participants, out, qa)
    figure_4(sensitivity, out, qa)
    figure_5(backbone, out, qa)
    figure_s1(classes, out, qa)
    figure_s2(sensitivity, out, qa)
    figure_s3(ablation, out, qa)

    (out / "figure_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    failures = [issue for item in qa for issue in item["issues"] if issue["severity"] == "FAIL"]
    print(f"Generated {len(qa)} figures; machine QA failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
