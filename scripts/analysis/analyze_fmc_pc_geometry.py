"""Audit FMC-PC axis-angle calibration against actual gravity-direction tilt.

This analysis is deliberately training-side only.  It reads the audited
subjects 1--10 placement profile JSON (which attests ``labels_read=false``),
reuses the exact rotation constructors used by training, and never loads a
REALDISP manifest, activity label, checkpoint, prediction, or test result.

Outputs include raw Monte-Carlo samples, ECDF values, descriptive statistics,
an audit JSON, and a publication-ready PDF/SVG/PNG figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

# On the Windows research environment, importing matplotlib before PyTorch
# avoids loading two OpenMP runtimes in the opposite order.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rcbd_har.engine import _axis_angle_rotation_matrices  # noqa: E402
from rcbd_har.placement import _random_axis_rotations  # noqa: E402


LIMBS = ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
ALLOWED_PARTICIPANTS = {f"subject{index}" for index in range(1, 11)}
COLORS = {
    "empirical": "#000000",
    "fmc": "#0072B2",
    "generic": "#D55E00",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_profile(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol", {})
    if protocol.get("labels_read") is not False:
        raise ValueError("Profile must explicitly attest labels_read=false")
    if protocol.get("target_validation_participants_read") is not False:
        raise ValueError("Profile must attest target/validation participants were not read")
    protocol_participants = {str(value) for value in protocol.get("participants", [])}
    if not protocol_participants or not protocol_participants <= ALLOWED_PARTICIPANTS:
        raise ValueError(
            "Geometry audit only permits training participants subject1--subject10"
        )

    rows: list[dict[str, Any]] = []
    for limb in LIMBS:
        profiles = payload.get("limbs", {}).get(limb, [])
        if not profiles:
            raise ValueError(f"No placement profiles found for {limb}")
        for profile in profiles:
            participant = str(profile.get("participant", ""))
            if participant not in ALLOWED_PARTICIPANTS:
                raise ValueError(f"Disallowed participant in profile: {participant}")
            angle = float(profile["angle_degrees"])
            if not math.isfinite(angle) or not 0.0 <= angle <= 180.0:
                raise ValueError(f"Invalid angle for {participant}/{limb}: {angle}")
            rows.append(
                {
                    "participant": participant,
                    "limb": limb,
                    "empirical_alpha_degrees": angle,
                }
            )
    frame = pd.DataFrame(rows).sort_values(["limb", "participant"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("Placement profile contains no usable angles")
    return payload, frame


def gravity_tilt_degrees(rotations: torch.Tensor) -> torch.Tensor:
    """Return angle between g=[0,0,1] and Rg for each proper rotation."""
    gravity = torch.zeros((rotations.shape[0], 3), dtype=rotations.dtype)
    gravity[:, 2] = 1.0
    rotated = torch.einsum("bij,bj->bi", rotations, gravity)
    cosine = (gravity * rotated).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def sample_distributions(
    profile: pd.DataFrame,
    samples_per_limb: int,
    seed: int,
) -> pd.DataFrame:
    """Sample exact training implementations with equal weight per sensor role."""
    torch.manual_seed(seed)
    sample_frames: list[pd.DataFrame] = []
    for limb in LIMBS:
        pool = torch.tensor(
            profile.loc[profile["limb"].eq(limb), "empirical_alpha_degrees"].to_numpy(),
            dtype=torch.float64,
        )
        indices = torch.randint(pool.numel(), (samples_per_limb,))
        fmc_alpha_degrees = pool[indices]
        fmc_rotations = _random_axis_rotations(
            torch.deg2rad(fmc_alpha_degrees), dtype=torch.float64
        )
        fmc_beta_degrees = gravity_tilt_degrees(fmc_rotations)

        # This is the exact generic baseline constructor: severity~U(0,1),
        # axis isotropic, and max_degrees=90.
        generic_severity = torch.rand(samples_per_limb, dtype=torch.float64)
        generic_rotations = _axis_angle_rotation_matrices(
            generic_severity, max_degrees=90.0, dtype=torch.float64
        )
        generic_alpha_degrees = 90.0 * generic_severity
        generic_beta_degrees = gravity_tilt_degrees(generic_rotations)

        sample_frames.extend(
            [
                pd.DataFrame(
                    {
                        "method": "FMC-PC random-axis",
                        "limb": limb,
                        "axis_angle_alpha_degrees": fmc_alpha_degrees.numpy(),
                        "actual_gravity_tilt_beta_degrees": fmc_beta_degrees.numpy(),
                    }
                ),
                pd.DataFrame(
                    {
                        "method": "Generic U(0,90) random-axis",
                        "limb": limb,
                        "axis_angle_alpha_degrees": generic_alpha_degrees.numpy(),
                        "actual_gravity_tilt_beta_degrees": generic_beta_degrees.numpy(),
                    }
                ),
            ]
        )
    return pd.concat(sample_frames, ignore_index=True)


def describe(values: np.ndarray) -> dict[str, float | int]:
    quantiles = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "n": int(values.size),
        "mean_degrees": float(np.mean(values)),
        "sd_degrees": float(np.std(values, ddof=1)),
        "q05_degrees": float(quantiles[0]),
        "q25_degrees": float(quantiles[1]),
        "median_degrees": float(quantiles[2]),
        "q75_degrees": float(quantiles[3]),
        "q95_degrees": float(quantiles[4]),
        "minimum_degrees": float(np.min(values)),
        "maximum_degrees": float(np.max(values)),
    }


def ecdf_at_grid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    ordered = np.sort(values)
    return np.searchsorted(ordered, grid, side="right") / ordered.size


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
        }
    )


def create_figure(
    profile: pd.DataFrame,
    samples: pd.DataFrame,
    output_stem: Path,
) -> None:
    configure_style()
    empirical = profile["empirical_alpha_degrees"].to_numpy(dtype=float)
    fmc = samples.loc[
        samples["method"].eq("FMC-PC random-axis"),
        "actual_gravity_tilt_beta_degrees",
    ].to_numpy(dtype=float)
    generic = samples.loc[
        samples["method"].eq("Generic U(0,90) random-axis"),
        "actual_gravity_tilt_beta_degrees",
    ].to_numpy(dtype=float)

    fig = plt.figure(figsize=(7.16, 3.05), constrained_layout=True)
    grid_spec = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0])
    ax_ecdf = fig.add_subplot(grid_spec[0, 0])
    ax_relation = fig.add_subplot(grid_spec[0, 1])

    curves = (
        (
            empirical,
            "Training Ideal-Self gap, alpha (n=36)",
            COLORS["empirical"],
            "-",
        ),
        (
            fmc,
            f"FMC-PC actual tilt, beta (n={len(fmc):,})",
            COLORS["fmc"],
            "-",
        ),
        (
            generic,
            f"Generic 0-90 deg actual tilt, beta (n={len(generic):,})",
            COLORS["generic"],
            "--",
        ),
    )
    for values, label, color, linestyle in curves:
        ordered = np.sort(values)
        cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
        ax_ecdf.step(
            ordered,
            cumulative,
            where="post",
            color=color,
            linestyle=linestyle,
            label=label,
            zorder=3,
        )
    # Raw profile points remain visible rather than being hidden by a smooth KDE.
    ax_ecdf.plot(
        empirical,
        np.full_like(empirical, -0.035),
        "|",
        color=COLORS["empirical"],
        markersize=4.0,
        markeredgewidth=0.7,
        clip_on=False,
    )
    ax_ecdf.set(xlim=(0, 180), ylim=(0, 1.02))
    ax_ecdf.set_xlabel("Angle (degrees)")
    ax_ecdf.set_ylabel("Empirical cumulative probability")
    ax_ecdf.set_title("a  Rotation-magnitude prior versus actual gravity tilt", loc="left")
    ax_ecdf.legend(loc="lower right", frameon=False, handlelength=2.6)
    ax_ecdf.grid(axis="y", color="#D9D9D9", linewidth=0.5, zorder=0)

    fmc_rows = samples[samples["method"].eq("FMC-PC random-axis")]
    # Deterministically thin to a readable point cloud while retaining the
    # complete Monte-Carlo sample in the CSV and ECDF.
    display = fmc_rows.iloc[:: max(1, len(fmc_rows) // 9000)].copy()
    ax_relation.scatter(
        display["axis_angle_alpha_degrees"],
        display["actual_gravity_tilt_beta_degrees"],
        s=3.5,
        alpha=0.10,
        linewidths=0,
        color=COLORS["fmc"],
        label="Random-axis samples",
    )
    ax_relation.plot(
        [0, 180],
        [0, 180],
        color="#333333",
        linestyle=":",
        linewidth=1.0,
        label="beta = alpha",
    )
    # Show conditional medians without implying a continuous trajectory.
    medians = (
        fmc_rows.groupby("axis_angle_alpha_degrees", as_index=False)[
            "actual_gravity_tilt_beta_degrees"
        ]
        .median()
        .sort_values("axis_angle_alpha_degrees")
    )
    ax_relation.scatter(
        medians["axis_angle_alpha_degrees"],
        medians["actual_gravity_tilt_beta_degrees"],
        s=12,
        marker="D",
        facecolor="white",
        edgecolor=COLORS["fmc"],
        linewidth=0.8,
        zorder=4,
        label="Conditional median",
    )
    ax_relation.set(xlim=(0, 180), ylim=(0, 180))
    ax_relation.set_aspect("equal", adjustable="box")
    ax_relation.set_xlabel("Sampled axis-angle magnitude, alpha (degrees)")
    ax_relation.set_ylabel("Actual gravity tilt, beta (degrees)")
    ax_relation.set_title("b  Random-axis geometry gives beta <= alpha", loc="left")
    ax_relation.legend(loc="upper left", frameon=False, markerscale=1.3)
    ax_relation.grid(color="#E5E5E5", linewidth=0.45, zorder=0)

    for axis in (ax_ecdf, ax_relation):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=3, width=0.7)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            output_stem.with_suffix(f".{suffix}"),
            dpi=600 if suffix == "png" else None,
        )

    # Gray-scale preview is an explicit accessibility audit artifact.
    png_path = output_stem.with_suffix(".png")
    from PIL import Image

    with Image.open(png_path) as image:
        image.convert("L").save(output_stem.with_name(output_stem.name + "_grayscale.png"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT
        / "results"
        / "fmc_pc_completion_20260721"
        / "placement_profiles_subjects1_10.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results"
        / "fmc_pc_revision_20260723_local"
        / "geometry",
    )
    parser.add_argument("--samples-per-limb", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    profile_path = args.profile.resolve()
    output_dir = args.output_dir.resolve()
    if args.samples_per_limb < 1_000:
        raise ValueError("Use at least 1,000 Monte-Carlo samples per limb")

    payload, profile = read_profile(profile_path)
    samples = sample_distributions(profile, args.samples_per_limb, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile.to_csv(output_dir / "training_empirical_alpha.csv", index=False)
    samples.to_csv(output_dir / "monte_carlo_rotation_samples.csv", index=False)

    empirical = profile["empirical_alpha_degrees"].to_numpy(dtype=float)
    fmc_rows = samples[samples["method"].eq("FMC-PC random-axis")]
    generic_rows = samples[samples["method"].eq("Generic U(0,90) random-axis")]
    fmc_alpha = fmc_rows["axis_angle_alpha_degrees"].to_numpy(dtype=float)
    fmc_beta = fmc_rows["actual_gravity_tilt_beta_degrees"].to_numpy(dtype=float)
    generic_beta = generic_rows["actual_gravity_tilt_beta_degrees"].to_numpy(dtype=float)

    grid = np.linspace(0.0, 180.0, 1801)
    ecdf = pd.DataFrame(
        {
            "angle_degrees": grid,
            "training_empirical_alpha_ecdf": ecdf_at_grid(empirical, grid),
            "fmc_pc_actual_beta_ecdf": ecdf_at_grid(fmc_beta, grid),
            "generic_actual_beta_ecdf": ecdf_at_grid(generic_beta, grid),
        }
    )
    ecdf.to_csv(output_dir / "distribution_ecdf.csv", index=False)

    summary_rows = []
    distributions = {
        "Training empirical alpha": empirical,
        "FMC-PC sampled alpha": fmc_alpha,
        "FMC-PC actual beta": fmc_beta,
        "Generic actual beta": generic_beta,
    }
    for name, values in distributions.items():
        summary_rows.append({"distribution": name, **describe(values)})
    pd.DataFrame(summary_rows).to_csv(output_dir / "distribution_summary.csv", index=False)

    alpha_radians = np.deg2rad(fmc_alpha)
    # Recover cos(theta_axis,g) algebraically from alpha and beta to verify
    # cos(beta)=cos(alpha)+(1-cos(alpha))*dot(axis,g)^2 within tolerance.
    # Direct matrix construction is the source of beta; the identity below is
    # checked using the invariant beta<=alpha because axes are not exported.
    beta_le_alpha_violation = np.maximum(fmc_beta - fmc_alpha, 0.0)
    generic_alpha = generic_rows["axis_angle_alpha_degrees"].to_numpy(dtype=float)
    generic_violation = np.maximum(generic_beta - generic_alpha, 0.0)

    comparisons = {
        "wasserstein_degrees_to_empirical_alpha": {
            "FMC-PC actual beta": float(wasserstein_distance(empirical, fmc_beta)),
            "Generic actual beta": float(wasserstein_distance(empirical, generic_beta)),
            "FMC-PC sampled alpha": float(wasserstein_distance(empirical, fmc_alpha)),
        },
        "two_sample_ks_to_empirical_alpha": {
            "FMC-PC actual beta": {
                "statistic": float(ks_2samp(empirical, fmc_beta).statistic),
                "pvalue_descriptive_only": float(ks_2samp(empirical, fmc_beta).pvalue),
            },
            "Generic actual beta": {
                "statistic": float(ks_2samp(empirical, generic_beta).statistic),
                "pvalue_descriptive_only": float(ks_2samp(empirical, generic_beta).pvalue),
            },
        },
        "geometry_invariants": {
            "fmc_max_beta_minus_alpha_degrees": float(beta_le_alpha_violation.max()),
            "generic_max_beta_minus_alpha_degrees": float(generic_violation.max()),
            "fmc_beta_le_alpha": bool(beta_le_alpha_violation.max() <= 1e-7),
            "generic_beta_le_alpha": bool(generic_violation.max() <= 1e-7),
            "finite_values": bool(
                np.isfinite(alpha_radians).all()
                and np.isfinite(fmc_beta).all()
                and np.isfinite(generic_beta).all()
            ),
        },
    }

    audit = {
        "analysis": "FMC-PC random-axis geometry audit",
        "training_only": True,
        "activity_labels_read": False,
        "manifests_read": False,
        "checkpoints_or_predictions_read": False,
        "test_participants_or_results_read": False,
        "permitted_profile_participants": sorted(ALLOWED_PARTICIPANTS),
        "profile_protocol": payload.get("protocol", {}),
        "profile_observations": int(len(profile)),
        "profile_exclusions": payload.get("excluded", []),
        "monte_carlo": {
            "seed": args.seed,
            "samples_per_limb_per_method": args.samples_per_limb,
            "limbs": list(LIMBS),
            "gravity_reference": [0.0, 0.0, 1.0],
            "invariance_note": (
                "For an isotropic random axis, the conditional beta distribution is "
                "independent of the chosen fixed unit gravity direction."
            ),
        },
        "geometry": {
            "definition": "beta = arccos(g_hat dot R(alpha,v) g_hat)",
            "identity": "cos(beta)=cos(alpha)+(1-cos(alpha))*(v dot g_hat)^2",
            "interpretation": (
                "FMC-PC calibrates the SO(3) axis-angle magnitude alpha; it does not "
                "force the realized gravity-direction tilt beta to equal alpha."
            ),
        },
        "source_sha256": {
            "profile": sha256(profile_path),
            "placement_py": sha256(ROOT / "src" / "rcbd_har" / "placement.py"),
            "engine_py": sha256(ROOT / "src" / "rcbd_har" / "engine.py"),
            "analysis_script": sha256(Path(__file__).resolve()),
        },
        "descriptive_comparisons": comparisons,
        "warning": (
            "KS p-values are descriptive only because the 36 empirical profile angles "
            "and Monte-Carlo draws do not represent equal independent biological samples."
        ),
    }
    (output_dir / "geometry_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    create_figure(profile, samples, output_dir / "fmc_pc_geometry_distribution")
    print(json.dumps(audit["descriptive_comparisons"], indent=2))
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
