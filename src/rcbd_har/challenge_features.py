from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy import stats


LIMB_ORDER = ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
AXES = ("x", "y", "z")
DEFAULT_SCALES_SECONDS = (1, 2, 4, 8, 16, 32)
TIME_FEATURE_NAMES = (
    "min",
    "max",
    "peak_to_peak",
    "iqr",
    "std",
    "skewness",
    "kurtosis",
    "hjorth_mobility",
    "hjorth_complexity",
    "mean_crossing_rate",
    "differential_entropy",
    "petrosian_fractal_dimension",
    "katz_fractal_dimension",
)


def repair_recording_from_validity(
    arrays: Mapping[str, np.ndarray],
    validity: Mapping[str, np.ndarray],
    strategy: str = "contralateral",
) -> dict[str, np.ndarray]:
    """Repair already-exported recordings while respecting their original validity masks."""
    if strategy not in {"zero", "contralateral"}:
        raise ValueError(f"Unknown missing-value strategy: {strategy}")
    repaired = {
        limb: np.asarray(arrays[limb], dtype=np.float32).copy() for limb in LIMB_ORDER
    }
    masks = {
        limb: np.asarray(validity.get(limb, np.isfinite(repaired[limb])), dtype=bool)
        for limb in LIMB_ORDER
    }
    for limb in LIMB_ORDER:
        if repaired[limb].shape != masks[limb].shape:
            raise ValueError(
                f"Validity shape mismatch for {limb}: {masks[limb].shape} vs "
                f"{repaired[limb].shape}"
            )
        masks[limb] &= np.isfinite(repaired[limb])
    if strategy == "contralateral":
        for left, right in (("left_wrist", "right_wrist"), ("left_ankle", "right_ankle")):
            left_source = repaired[left].copy()
            right_source = repaired[right].copy()
            fill_left = ~masks[left] & masks[right]
            fill_right = ~masks[right] & masks[left]
            repaired[left][fill_left] = right_source[fill_left]
            repaired[right][fill_right] = left_source[fill_right]
            masks[left][fill_left] = True
            masks[right][fill_right] = True
    for limb in LIMB_ORDER:
        repaired[limb][~masks[limb]] = 0.0
        repaired[limb] = np.nan_to_num(repaired[limb], copy=False)
    return repaired


def features_per_axis(scales_seconds: Sequence[int] = DEFAULT_SCALES_SECONDS) -> int:
    return 2 * sum(len(TIME_FEATURE_NAMES) + (int(scale) > 1) for scale in scales_seconds)


def feature_names(scales_seconds: Sequence[int] = DEFAULT_SCALES_SECONDS) -> list[str]:
    names: list[str] = []
    for limb in LIMB_ORDER:
        for axis in AXES:
            for direction in ("past", "future"):
                for scale in scales_seconds:
                    current = list(TIME_FEATURE_NAMES)
                    if int(scale) > 1:
                        current.append("spectral_entropy")
                    names.extend(
                        f"{limb}__{axis}__{direction}_{int(scale)}s__{name}" for name in current
                    )
    return names


def _feature_block(windows: np.ndarray, sample_rate: float, include_spectral: bool) -> np.ndarray:
    """Compute the public WEAR Challenge feature family for [batch,time,axis] windows."""
    import antropy as ant

    values = np.moveaxis(np.asarray(windows, dtype=np.float64), 1, -1)  # [B,C,T]
    centered = values - values.mean(axis=-1, keepdims=True)
    mobility, complexity = ant.hjorth_params(values, axis=-1)
    crossings = np.mean(centered[..., :-1] * centered[..., 1:] < 0, axis=-1)
    columns = [
        np.min(values, axis=-1),
        np.max(values, axis=-1),
        np.ptp(values, axis=-1),
        stats.iqr(values, axis=-1),
        np.std(values, axis=-1),
        stats.skew(values, axis=-1, bias=True),
        stats.kurtosis(values, axis=-1, fisher=True, bias=True),
        mobility,
        complexity,
        crossings,
        stats.differential_entropy(values, axis=-1),
        ant.petrosian_fd(values, axis=-1),
        ant.katz_fd(values, axis=-1),
    ]
    if include_spectral:
        columns.append(
            ant.spectral_entropy(
                values, sf=float(sample_rate), method="welch", normalize=True, axis=-1
            )
        )
    output = np.stack(columns, axis=-1)  # [B,C,F]
    return np.nan_to_num(output, nan=np.nan, posinf=np.nan, neginf=np.nan).astype(np.float32)


def extract_multiresolution_features(
    arrays: Mapping[str, np.ndarray],
    centers: np.ndarray,
    sample_rate: float = 50.0,
    scales_seconds: Sequence[int] = DEFAULT_SCALES_SECONDS,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract past/future multi-resolution features without crossing recording boundaries."""
    missing = set(LIMB_ORDER).difference(arrays)
    if missing:
        raise KeyError(f"Missing limb arrays: {sorted(missing)}")
    ordered = [np.asarray(arrays[limb], dtype=np.float32) for limb in LIMB_ORDER]
    shapes = {array.shape for array in ordered}
    if len(shapes) != 1 or ordered[0].ndim != 2 or ordered[0].shape[1] != 3:
        raise ValueError(f"Expected four equal [samples,3] arrays, got {[x.shape for x in ordered]}")
    if sample_rate <= 0 or batch_size <= 0:
        raise ValueError("sample_rate and batch_size must be positive")
    centers = np.asarray(centers, dtype=np.int64)
    recording = np.concatenate(ordered, axis=1)  # limb-major: 12 axes
    channel_blocks: list[list[np.ndarray]] = [[] for _ in range(recording.shape[1])]
    for direction in ("past", "future"):
        for scale in scales_seconds:
            width = int(round(float(scale) * sample_rate))
            if width < 2:
                raise ValueError(f"Scale {scale}s yields an invalid {width}-sample window")
            block = np.full(
                (
                    len(centers),
                    recording.shape[1],
                    len(TIME_FEATURE_NAMES) + (int(scale) > 1),
                ),
                np.nan,
                dtype=np.float32,
            )
            if direction == "past":
                valid_rows = np.flatnonzero((centers - width >= 0) & (centers <= len(recording)))
                starts = centers[valid_rows] - width
            else:
                valid_rows = np.flatnonzero((centers >= 0) & (centers + width <= len(recording)))
                starts = centers[valid_rows]
            for begin in range(0, len(valid_rows), batch_size):
                rows = valid_rows[begin : begin + batch_size]
                batch_starts = starts[begin : begin + batch_size]
                windows = np.stack(
                    [recording[start : start + width] for start in batch_starts], axis=0
                )
                block[rows] = _feature_block(
                    windows, sample_rate=sample_rate, include_spectral=int(scale) > 1
                )
            for channel in range(recording.shape[1]):
                channel_blocks[channel].append(block[:, channel])
    blocks = [np.concatenate(parts, axis=1) for parts in channel_blocks]
    output = np.concatenate(blocks, axis=1)
    expected = len(feature_names(scales_seconds))
    if output.shape != (len(centers), expected):
        raise RuntimeError(f"Feature shape mismatch: {output.shape}, expected {(len(centers), expected)}")
    return output
