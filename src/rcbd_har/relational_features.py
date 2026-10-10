from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from rcbd_har.challenge_features import LIMB_ORDER
from rcbd_har.ul_pairing import UL_PAIRS


DEFAULT_RELATIONAL_SCALES_SECONDS = (1, 2, 4, 8, 16, 32)
LIMB_FEATURE_NAMES = (
    "magnitude_mean",
    "magnitude_std",
    "dynamic_rms",
    "magnitude_peak_to_peak",
    "signal_magnitude_area",
    "jerk_rms",
    "log_spectral_power",
    "dominant_frequency",
    "spectral_centroid",
    "spectral_entropy",
    "bandpower_0p3_1hz",
    "bandpower_1_3hz",
    "bandpower_3_6hz",
    "bandpower_6_12p5hz",
    "periodicity",
)
PAIR_FEATURE_NAMES = (
    "magnitude_correlation",
    "jerk_correlation",
    "log_dynamic_rms_ratio",
    "dominant_frequency_difference",
    "spectral_cosine_similarity",
    "log_spectral_power_ratio",
    "mean_orientation_cosine",
    "envelope_correlation",
)


def relational_feature_names(
    scales_seconds: Sequence[int] = DEFAULT_RELATIONAL_SCALES_SECONDS,
) -> tuple[list[str], list[str]]:
    limb_names = [
        f"{direction}_{int(scale)}s__{name}"
        for direction in ("past", "future")
        for scale in scales_seconds
        for name in LIMB_FEATURE_NAMES
    ]
    pair_names = [
        f"{direction}_{int(scale)}s__{name}"
        for direction in ("past", "future")
        for scale in scales_seconds
        for name in PAIR_FEATURE_NAMES
    ]
    return limb_names, pair_names


def _safe_correlation(left: np.ndarray, right: np.ndarray, axis: int) -> np.ndarray:
    left_centered = left - left.mean(axis=axis, keepdims=True)
    right_centered = right - right.mean(axis=axis, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=axis)
    denominator = np.sqrt(
        np.sum(left_centered**2, axis=axis) * np.sum(right_centered**2, axis=axis)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 1e-12,
    )


def _window_features(
    windows: np.ndarray, sample_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return limb [B,4,F] and UL-pair [B,4,R] features for raw [B,T,4,3]."""
    values = np.asarray(windows, dtype=np.float64)
    magnitude = np.linalg.norm(values, axis=-1)
    dynamic = magnitude - magnitude.mean(axis=1, keepdims=True)
    jerk = np.diff(magnitude, axis=1) * float(sample_rate)
    axis_dynamic = values - values.mean(axis=1, keepdims=True)

    spectrum = np.fft.rfft(dynamic, axis=1)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(values.shape[1], d=1.0 / float(sample_rate))
    positive = frequencies > 0
    positive_power = power[:, positive, :]
    positive_frequencies = frequencies[positive]
    total_power = positive_power.sum(axis=1)
    normalized_power = np.divide(
        positive_power,
        total_power[:, None, :] + 1e-12,
    )
    dominant = positive_frequencies[np.argmax(positive_power, axis=1)]
    centroid = np.sum(normalized_power * positive_frequencies[None, :, None], axis=1)
    entropy = -np.sum(normalized_power * np.log(normalized_power + 1e-12), axis=1)
    entropy /= np.log(max(2, len(positive_frequencies)))

    bands = ((0.3, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, 12.5))
    band_features = []
    for low, high in bands:
        mask = (positive_frequencies >= low) & (positive_frequencies < high)
        if np.any(mask):
            band_features.append(normalized_power[:, mask, :].sum(axis=1))
        else:
            band_features.append(np.zeros_like(total_power))

    autocorrelation = np.fft.irfft(power, n=values.shape[1], axis=1)
    lag_start = max(1, int(round(0.25 * sample_rate)))
    lag_stop = min(values.shape[1], int(round(3.0 * sample_rate)) + 1)
    if lag_start < lag_stop:
        periodicity = np.max(autocorrelation[:, lag_start:lag_stop, :], axis=1)
        periodicity = np.divide(
            periodicity,
            autocorrelation[:, 0, :] + 1e-12,
            out=np.zeros_like(periodicity),
            where=autocorrelation[:, 0, :] > 1e-12,
        )
    else:
        periodicity = np.zeros_like(total_power)

    dynamic_rms = np.sqrt(np.mean(dynamic**2, axis=1))
    jerk_rms = np.sqrt(np.mean(jerk**2, axis=1))
    limb = np.stack(
        [
            magnitude.mean(axis=1),
            magnitude.std(axis=1),
            dynamic_rms,
            np.ptp(magnitude, axis=1),
            np.mean(np.sum(np.abs(axis_dynamic), axis=-1), axis=1),
            jerk_rms,
            np.log1p(total_power),
            dominant,
            centroid,
            entropy,
            *band_features,
            periodicity,
        ],
        axis=-1,
    )

    mean_vectors = values.mean(axis=1)
    mean_norms = np.linalg.norm(mean_vectors, axis=-1)
    chunk_count = min(8, values.shape[1])
    usable = (values.shape[1] // chunk_count) * chunk_count
    envelope = magnitude[:, :usable].reshape(
        values.shape[0], chunk_count, usable // chunk_count, len(LIMB_ORDER)
    ).mean(axis=2)
    pair_columns: list[np.ndarray] = []
    limb_index = {name: index for index, name in enumerate(LIMB_ORDER)}
    for wrist_name, ankle_name in UL_PAIRS:
        wrist = limb_index[wrist_name]
        ankle = limb_index[ankle_name]
        spectral_numerator = np.sum(
            normalized_power[:, :, wrist] * normalized_power[:, :, ankle], axis=1
        )
        spectral_denominator = np.sqrt(
            np.sum(normalized_power[:, :, wrist] ** 2, axis=1)
            * np.sum(normalized_power[:, :, ankle] ** 2, axis=1)
        )
        orientation_denominator = mean_norms[:, wrist] * mean_norms[:, ankle]
        pair_columns.append(
            np.stack(
                [
                    _safe_correlation(dynamic[:, :, wrist], dynamic[:, :, ankle], axis=1),
                    _safe_correlation(jerk[:, :, wrist], jerk[:, :, ankle], axis=1),
                    np.log((dynamic_rms[:, wrist] + 1e-6) / (dynamic_rms[:, ankle] + 1e-6)),
                    np.abs(dominant[:, wrist] - dominant[:, ankle]),
                    np.divide(
                        spectral_numerator,
                        spectral_denominator,
                        out=np.zeros_like(spectral_numerator),
                        where=spectral_denominator > 1e-12,
                    ),
                    np.log((total_power[:, wrist] + 1e-6) / (total_power[:, ankle] + 1e-6)),
                    np.divide(
                        np.sum(mean_vectors[:, wrist] * mean_vectors[:, ankle], axis=1),
                        orientation_denominator,
                        out=np.zeros_like(orientation_denominator),
                        where=orientation_denominator > 1e-12,
                    ),
                    _safe_correlation(envelope[:, :, wrist], envelope[:, :, ankle], axis=1),
                ],
                axis=-1,
            )
        )
    pair = np.stack(pair_columns, axis=1)
    return limb.astype(np.float32), pair.astype(np.float32)


def extract_relational_features(
    arrays: Mapping[str, np.ndarray],
    centers: np.ndarray,
    sample_rate: float = 50.0,
    scales_seconds: Sequence[int] = DEFAULT_RELATIONAL_SCALES_SECONDS,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    ordered = [np.asarray(arrays[limb], dtype=np.float32) for limb in LIMB_ORDER]
    shapes = {value.shape for value in ordered}
    if len(shapes) != 1 or ordered[0].ndim != 2 or ordered[0].shape[1] != 3:
        raise ValueError(f"Expected four equal [samples,3] arrays, got {[x.shape for x in ordered]}")
    recording = np.stack(ordered, axis=1)  # [T,4,3]
    centers = np.asarray(centers, dtype=np.int64)
    limb_parts: list[np.ndarray] = []
    pair_parts: list[np.ndarray] = []
    for direction in ("past", "future"):
        for scale in scales_seconds:
            width = int(round(float(scale) * sample_rate))
            limb_block = np.full(
                (len(centers), len(LIMB_ORDER), len(LIMB_FEATURE_NAMES)), np.nan, np.float32
            )
            pair_block = np.full(
                (len(centers), len(UL_PAIRS), len(PAIR_FEATURE_NAMES)), np.nan, np.float32
            )
            if direction == "past":
                rows = np.flatnonzero((centers - width >= 0) & (centers <= len(recording)))
                starts = centers[rows] - width
            else:
                rows = np.flatnonzero((centers >= 0) & (centers + width <= len(recording)))
                starts = centers[rows]
            for begin in range(0, len(rows), batch_size):
                selected = rows[begin : begin + batch_size]
                selected_starts = starts[begin : begin + batch_size]
                windows = np.stack(
                    [recording[start : start + width] for start in selected_starts], axis=0
                )
                limb_block[selected], pair_block[selected] = _window_features(windows, sample_rate)
            limb_parts.append(limb_block)
            pair_parts.append(pair_block)
    limb_features = np.concatenate(limb_parts, axis=-1)
    pair_features = np.concatenate(pair_parts, axis=-1)
    limb_names, pair_names = relational_feature_names(scales_seconds)
    if limb_features.shape != (len(centers), len(LIMB_ORDER), len(limb_names)):
        raise RuntimeError(f"Unexpected limb feature shape: {limb_features.shape}")
    if pair_features.shape != (len(centers), len(UL_PAIRS), len(pair_names)):
        raise RuntimeError(f"Unexpected pair feature shape: {pair_features.shape}")
    return limb_features, pair_features


def make_ul_relational_variant(
    base_features: np.ndarray,
    limb_features: np.ndarray,
    pair_features: np.ndarray,
    pair_index: int,
    feature_set: str = "full",
) -> np.ndarray:
    from rcbd_har.ul_pairing import select_pair

    wrist_name, ankle_name = UL_PAIRS[pair_index]
    limb_index = {name: index for index, name in enumerate(LIMB_ORDER)}
    if feature_set not in {"full", "limb", "pair"}:
        raise ValueError(f"Unknown relational feature set: {feature_set}")
    blocks = [select_pair(base_features, UL_PAIRS[pair_index])]
    if feature_set in {"full", "limb"}:
        blocks.extend(
            [
                limb_features[:, limb_index[wrist_name]],
                limb_features[:, limb_index[ankle_name]],
            ]
        )
    if feature_set in {"full", "pair"}:
        blocks.append(pair_features[:, pair_index])
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
