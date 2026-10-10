"""Generate frozen profile variants for the REALDISP mechanism controls.

Two amplitude-source variants are derived from the frozen empirical placement
profile (SHA-256 b05eedb46aa01cff0c2331b09bc3ca4911821a0ef7c8291c0807dd19c555b2b7):

- ``placement_profiles_beta_moment_matched.json``: every record angle is replaced
  by an independent draw from a scaled Beta distribution on [0, A_max] matched
  to the empirical mean and population SD, where A_max is the maximum empirical
  angle.  Supports range and moments match the empirical pool; the empirical
  frequency shape does not.
- ``placement_profiles_clip90.json``: every record angle above 90 degrees is
  clipped to exactly 90 degrees; all other records stay unchanged.

Rotation matrices are rebuilt with Rodrigues formula from the ORIGINAL record
axis and the NEW angle, because ``load_placement_profile_bank`` derives the
amplitude from the matrix trace rather than the ``angle_degrees`` field and
enforces orthogonality/properness.  All other record fields are preserved.

The script is deterministic (fixed RNG seed) and writes a sidecar manifest with
SHA-256 values for the runner freeze checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

FROZEN_PROFILE_SHA256 = (
    "b05eedb46aa01cff0c2331b09bc3ca4911821a0ef7c8291c0807dd19c555b2b7"
)
LIMB_ORDER = ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
VARIANT_SEED = 20260920
CLIP_THRESHOLD_DEGREES = 90.0


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    found = sha256_of(path)
    if found != FROZEN_PROFILE_SHA256:
        raise ValueError(
            f"Frozen profile SHA mismatch: expected {FROZEN_PROFILE_SHA256}, found {found}"
        )
    return payload


def empirical_stats(payload: dict) -> tuple[list[float], float, float, float]:
    values: list[float] = []
    for limb in LIMB_ORDER:
        values.extend(float(record["angle_degrees"]) for record in payload["limbs"][limb])
    array = np.asarray(values, dtype=np.float64)
    # ddof=1 matches the convention of data/geometry/distribution_summary.csv
    # (training empirical alpha: mean 40.8407, SD 46.1240), so the control
    # targets the same published moments.
    return values, float(array.mean()), float(array.std(ddof=1)), float(array.max())


def rotation_from_axis_angle(axis: np.ndarray, degrees: float) -> list[list[float]]:
    unit = axis / np.linalg.norm(axis)
    rad = math.radians(degrees)
    cos, sin = math.cos(rad), math.sin(rad)
    cross = np.array(
        [
            [0.0, -unit[2], unit[1]],
            [unit[2], 0.0, -unit[0]],
            [-unit[1], unit[0], 0.0],
        ]
    )
    outer = np.outer(unit, unit)
    matrix = cos * np.eye(3) + (1.0 - cos) * outer + sin * cross
    return [[float(value) for value in row] for row in matrix]


def axis_from_rotation(matrix: list[list[float]], degrees: float) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    rad = math.radians(degrees)
    skew = (array - array.T) / (2.0 * math.sin(rad))
    axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    norm = float(np.linalg.norm(axis))
    if not math.isclose(norm, 1.0, abs_tol=1e-6):
        raise ValueError(f"Rotation axis extraction failed: norm={norm}")
    return axis


def angle_from_rotation(matrix: list[list[float]]) -> float:
    trace = float(np.trace(np.asarray(matrix, dtype=np.float64)))
    return math.degrees(math.acos(((trace - 1.0) / 2.0).__round__(12)))


def clone_with_new_angles(
    frozen: dict, new_angles: dict[tuple[str, int], float]
) -> dict:
    variant = json.loads(json.dumps(frozen))  # deep copy
    for limb in LIMB_ORDER:
        records = variant["limbs"][limb]
        order = sorted(
            range(len(records)),
            key=lambda index: int(records[index]["participant"].replace("subject", "")),
        )
        for position, record_index in enumerate(order):
            key = (limb, record_index)
            if key not in new_angles:
                continue
            record = records[record_index]
            old_angle = float(record["angle_degrees"])
            new_angle = new_angles[key]
            axis = axis_from_rotation(record["rotation"], old_angle)
            record["rotation"] = rotation_from_axis_angle(axis, new_angle)
            record["angle_degrees"] = new_angle
            rebuilt = angle_from_rotation(record["rotation"])
            if abs(rebuilt - new_angle) > 1e-6:
                raise ValueError(
                    f"Rebuild check failed for {limb} record {record_index}: "
                    f"{rebuilt} != {new_angle}"
                )
    return variant


MIN_LIMB_ANGLE_FLOOR_DEGREES = 0.5


def _limb_beta_quantiles(angles):
    """Moment-matched scaled-Beta stratified quantile grid for ONE limb.

    Support [0, A], mean and population SD are all derived from this limb own
    empirical records, so every limb keeps its amplitude marginal including
    its tail. A single global Beta whose sorted grid is sliced across limbs
    was shown to strip tail coverage from the wrists.
    """
    empirical = np.asarray(angles, dtype=np.float64)
    count = int(empirical.size)
    support_max = float(empirical.max())
    target_mean = float(empirical.mean())
    target_sd = float(empirical.std(ddof=0))
    info = {
        "records": count,
        "support_max_degrees": support_max,
        "target_mean_degrees": target_mean,
        "target_sd_population_degrees": target_sd,
        "variance_clamped_to_support": False,
        "degenerate": False,
        "quantiles_floored": 0,
    }
    if support_max <= 0.0 or not (0.0 < target_mean < support_max):
        info["degenerate"] = True
        info.update(alpha=0.0, beta=0.0,
                    realized_mean_degrees=target_mean,
                    realized_sd_population_degrees=target_sd,
                    realized_max_degrees=support_max)
        return empirical.copy(), info
    unit_mean = target_mean / support_max
    unit_var = (target_sd / support_max) ** 2.0
    variance_cap = unit_mean * (1.0 - unit_mean)
    if not (0.0 < unit_var < variance_cap):
        unit_var = 0.99 * variance_cap
        info["variance_clamped_to_support"] = True
    total = unit_mean * (1.0 - unit_mean) / unit_var - 1.0
    alpha = unit_mean * total
    beta = (1.0 - unit_mean) * total
    probabilities = (np.arange(count, dtype=np.float64) + 0.5) / count
    draws = stats.beta.ppf(probabilities, alpha, beta) * support_max
    # Near-zero angles make axis recovery (R-R^T)/(2 sin theta) ill-conditioned;
    # enforce a tiny strictly-increasing floor on the left tail (negligible moments).
    previous = 0.0
    floored = 0
    for k in range(count):
        lower = MIN_LIMB_ANGLE_FLOOR_DEGREES if k == 0 else previous + 0.05
        if draws[k] < lower:
            draws[k] = lower
            floored += 1
        previous = float(draws[k])
    info["quantiles_floored"] = floored
    info["min_angle_floor_degrees"] = MIN_LIMB_ANGLE_FLOOR_DEGREES
    if not np.all(np.isfinite(draws)) or np.any(draws <= 0.0) or np.any(draws > support_max + 1e-9):
        raise ValueError("Per-limb Beta quantile draws left the open support interval")
    info.update(
        alpha=float(alpha),
        beta=float(beta),
        realized_mean_degrees=float(draws.mean()),
        realized_sd_population_degrees=float(draws.std(ddof=0)),
        realized_max_degrees=float(draws.max()),
    )
    return draws, info


def beta_moment_matched_variant(frozen: dict) -> tuple[dict, dict]:
    assignment: dict[tuple[str, int], float] = {}
    draw_log: list[dict[str, object]] = []
    limb_report: dict = {}
    for limb in LIMB_ORDER:
        records = frozen["limbs"][limb]
        order = sorted(
            range(len(records)),
            key=lambda index: int(records[index]["participant"].replace("subject", "")),
        )
        empirical = np.asarray(
            [float(records[index]["angle_degrees"]) for index in order],
            dtype=np.float64,
        )
        draws, info = _limb_beta_quantiles(empirical)
        limb_report[limb] = info
        for position, record_index in enumerate(order):
            angle = float(draws[position])
            participant = records[record_index]["participant"]
            assignment[(limb, record_index)] = angle
            draw_log.append(
                {
                    "limb": limb,
                    "participant": participant,
                    "empirical_degrees": float(records[record_index]["angle_degrees"]),
                    "beta_degrees": angle,
                }
            )
    variant = clone_with_new_angles(frozen, assignment)
    realized = np.asarray([float(value) for value in assignment.values()], dtype=np.float64)
    _, empirical_mean, empirical_sample_sd, empirical_max = empirical_stats(frozen)
    variant["protocol"] = {
        **variant["protocol"],
        "variant": "beta_moment_matched_amplitude",
        "variant_method": "per_limb_beta_moment_quantiles",
        "variant_seed": VARIANT_SEED,
        "variant_rng": "scipy.stats.beta.ppf stratified quantiles, independent per limb",
        "support_max_degrees": float(empirical_max),
        "target_mean_degrees": float(empirical_mean),
        "target_sd_sample_degrees": float(empirical_sample_sd),
        "realized_mean_degrees": float(realized.mean()),
        "realized_sd_sample_degrees": float(realized.std(ddof=1)),
        "per_limb": limb_report,
        "note": (
            "Amplitude-only control with INDEPENDENT per-limb scaled-Beta moment "
            "matching: each limb support, mean and SD are matched from that limb "
            "own empirical records and a stratified quantile grid is drawn within "
            "the limb, so every limb (wrists included) retains its own tail "
            "coverage. Record axes, gain and noise are unchanged. A 0.5 degree "
            "strictly-increasing floor guards rotation-axis numerical stability."
        ),
    }
    audit = {
        "method": "per_limb_beta_moment_quantiles",
        "limbs": limb_report,
        "draws": draw_log,
    }
    return variant, audit


def clip_variant(frozen: dict) -> tuple[dict, dict]:
    assignment: dict[tuple[str, int], float] = {}
    changes: list[dict[str, object]] = []
    for limb in LIMB_ORDER:
        records = frozen["limbs"][limb]
        for record_index, record in enumerate(records):
            angle = float(record["angle_degrees"])
            if angle > CLIP_THRESHOLD_DEGREES:
                assignment[(limb, record_index)] = CLIP_THRESHOLD_DEGREES
                changes.append(
                    {
                        "limb": limb,
                        "participant": record["participant"],
                        "original_degrees": angle,
                        "clipped_degrees": CLIP_THRESHOLD_DEGREES,
                    }
                )
    if len(changes) != 5:
        raise ValueError(f"Expected 5 clipped records, found {len(changes)}")
    variant = clone_with_new_angles(frozen, assignment)
    variant["protocol"] = {
        **variant["protocol"],
        "variant": "empirical_amplitude_clip90",
        "note": (
            "Amplitude-only control: record angles above 90 degrees are clipped to "
            "exactly 90 degrees; axes and all other fields unchanged."
        ),
    }
    audit = {"changed_records": changes}
    return variant, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-profile", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    frozen = load_frozen(args.frozen_profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    for name, builder in (
        ("placement_profiles_beta_moment_matched.json", beta_moment_matched_variant),
        ("placement_profiles_clip90.json", clip_variant),
    ):
        variant, audit = builder(frozen)
        destination = args.output_dir / name
        destination.write_text(
            json.dumps(variant, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        audit_path = destination.with_suffix(".audit.json")
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        outputs[name] = sha256_of(destination)
        print(f"{name}: sha256={outputs[name]}")
    print(f"frozen: sha256={FROZEN_PROFILE_SHA256}")


if __name__ == "__main__":
    main()
