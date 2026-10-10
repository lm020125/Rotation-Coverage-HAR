from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image


SKILL_ROOT = Path(r"C:\Users\20921\.codex\skills\scipilot-figure-skill")
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from export_figure import export_figure  # noqa: E402
from layout_tools import add_panel_labels, finalize_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report, render_preview  # noqa: E402


LIMB_ORDER = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]
LIMB_LABELS = {
    "left_wrist": "Left wrist",
    "right_wrist": "Right wrist",
    "left_ankle": "Left ankle",
    "right_ankle": "Right ankle",
}
OKABE = ["#0072B2", "#56B4E9", "#D55E00", "#E69F00"]


def load_profiles(path: Path) -> tuple[pd.DataFrame, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for limb, profiles in payload["limbs"].items():
        for profile in profiles:
            rows.append(
                {
                    "participant": profile["participant"],
                    "limb": limb,
                    "angle_degrees": float(profile["angle_degrees"]),
                    "gain": float(profile["gain"]),
                    "raw_gain": float(profile["raw_gain"]),
                    "noise_std": float(profile["noise_std"]),
                    "ideal_zero_fraction": float(profile["ideal_zero_fraction"]),
                    "self_zero_fraction": float(profile["self_zero_fraction"]),
                    "ideal_valid_samples": int(profile["ideal_valid_samples"]),
                    "self_valid_samples": int(profile["self_valid_samples"]),
                }
            )
    frame = pd.DataFrame(rows)
    frame["limb"] = pd.Categorical(frame["limb"], LIMB_ORDER, ordered=True)
    frame = frame.sort_values(["limb", "participant"]).reset_index(drop=True)
    return frame, payload


def build_summary(frame: pd.DataFrame, payload: dict) -> dict:
    expected_participants = list(payload["protocol"]["participants"])
    expected_profiles = len(expected_participants) * len(LIMB_ORDER)
    observed_participants = sorted(frame["participant"].unique().tolist())
    missing_participants = sorted(set(expected_participants) - set(observed_participants))
    per_limb = {}
    for limb in LIMB_ORDER:
        subset = frame[frame["limb"] == limb]
        per_limb[limb] = {
            "n": int(len(subset)),
            "angle_degrees_mean": float(subset["angle_degrees"].mean()),
            "angle_degrees_min": float(subset["angle_degrees"].min()),
            "angle_degrees_max": float(subset["angle_degrees"].max()),
            "gain_mean": float(subset["gain"].mean()),
            "noise_std_mean": float(subset["noise_std"].mean()),
        }
    return {
        "source": str(payload["protocol"]["name"]),
        "labels_read": bool(payload["protocol"]["labels_read"]),
        "expected_profiles": expected_profiles,
        "observed_profiles": int(len(frame)),
        "observed_participants": observed_participants,
        "missing_profile_participants": missing_participants,
        "angle_degrees": {
            "mean": float(frame["angle_degrees"].mean()),
            "min": float(frame["angle_degrees"].min()),
            "max": float(frame["angle_degrees"].max()),
            "above_90_count": int((frame["angle_degrees"] > 90).sum()),
        },
        "gain": {
            "mean": float(frame["gain"].mean()),
            "min": float(frame["gain"].min()),
            "max": float(frame["gain"].max()),
            "clipped_low_count": int(np.isclose(frame["gain"], 0.75).sum()),
            "clipped_high_count": int(np.isclose(frame["gain"], 1.25).sum()),
        },
        "noise_std": {
            "mean": float(frame["noise_std"].mean()),
            "min": float(frame["noise_std"].min()),
            "max": float(frame["noise_std"].max()),
            "nonzero_count": int((frame["noise_std"] > 0).sum()),
        },
        "per_limb": per_limb,
    }


def strip_panel(ax, frame: pd.DataFrame, y: str, ylabel: str) -> None:
    sns.stripplot(
        data=frame,
        x="limb",
        y=y,
        order=LIMB_ORDER,
        hue="limb",
        palette=dict(zip(LIMB_ORDER, OKABE, strict=True)),
        jitter=0.16,
        size=4.2,
        alpha=0.86,
        edgecolor="black",
        linewidth=0.35,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(LIMB_ORDER)), [LIMB_LABELS[x] for x in LIMB_ORDER])
    ax.tick_params(axis="x", rotation=22)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, linestyle="--")
    sns.despine(ax=ax)


def build_figure(frame: pd.DataFrame, participants: list[str], output_dir: Path) -> None:
    setup_style(journal="nature", lang="en")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.15), constrained_layout=True)

    strip_panel(axes[0, 0], frame, "angle_degrees", "Gravity-identifiable tilt (degrees)")
    axes[0, 0].axhline(90, color="#555555", linestyle="--", linewidth=0.75)
    axes[0, 0].text(
        3.45,
        92.5,
        "90-degree generic-control limit",
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#444444",
    )

    strip_panel(axes[0, 1], frame, "gain", "RMS gain ratio")
    axes[0, 1].axhline(0.75, color="#777777", linestyle=":", linewidth=0.7)
    axes[0, 1].axhline(1.25, color="#777777", linestyle=":", linewidth=0.7)
    axes[0, 1].set_ylim(0.72, 1.28)
    axes[0, 1].text(
        0.99,
        0.94,
        "Clipping bounds: 0.75-1.25",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color="#444444",
    )

    strip_panel(axes[1, 0], frame, "noise_std", "Excess first-difference noise (SD)")
    axes[1, 0].set_ylim(bottom=-0.006)

    availability = (
        frame.assign(value=1)
        .pivot_table(index="participant", columns="limb", values="value", aggfunc="max", observed=False)
        .reindex(index=participants, columns=LIMB_ORDER)
        .fillna(0)
        .astype(int)
    )
    cmap = ListedColormap(["#D9D9D9", "#0072B2"])
    sns.heatmap(
        availability,
        cmap=cmap,
        vmin=0,
        vmax=1,
        cbar=False,
        linewidths=0.7,
        linecolor="white",
        square=False,
        ax=axes[1, 1],
    )
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("")
    axes[1, 1].set_xticklabels([LIMB_LABELS[x] for x in LIMB_ORDER], rotation=22, ha="right")
    axes[1, 1].set_yticklabels([x.replace("subject", "S") for x in participants], rotation=0)
    axes[1, 1].legend(
        handles=[
            Patch(facecolor="#0072B2", label="Profile available"),
            Patch(facecolor="#D9D9D9", label="Rejected / unavailable"),
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=6.2,
    )

    finalize_figure(fig)
    add_panel_labels(fig, style="nature")
    output_dir.mkdir(parents=True, exist_ok=True)
    render_preview(fig, output_dir / "_preview_figureS5_profile_bank_diagnostics.png", dpi=180)
    print_report(audit_layout(fig))
    export_figure(
        fig,
        output_dir / "figureS5_profile_bank_diagnostics",
        formats=["pdf", "svg", "png"],
        size_inches=(7.2, 5.15),
        dpi=600,
        grayscale_preview=False,
        tight=False,
    )
    Image.open(output_dir / "figureS5_profile_bank_diagnostics.png").convert("L").save(
        output_dir / "figureS5_profile_bank_diagnostics_grayscale.png"
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build REALDISP placement-profile diagnostics.")
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()

    frame, payload = load_profiles(args.profiles)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.data_dir / "profile_bank_diagnostics.csv", index=False)
    summary = build_summary(frame, payload)
    (args.data_dir / "profile_bank_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if not args.data_only:
        build_figure(frame, list(payload["protocol"]["participants"]), args.figure_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
