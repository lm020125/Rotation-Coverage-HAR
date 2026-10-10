from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ACTIVITY_TO_LABEL = {
    "jogging": 1,
    "jogging (rotating arms)": 2,
    "jogging (skipping)": 3,
    "jogging (sidesteps)": 4,
    "jogging (butt-kicks)": 5,
    "stretching (triceps)": 6,
    "stretching (lunging)": 7,
    "stretching (shoulders)": 8,
    "stretching (hamstrings)": 9,
    "stretching (lumbar rotation)": 10,
    "push-ups": 11,
    "push-ups (complex)": 12,
    "sit-ups": 13,
    "sit-ups (complex)": 14,
    "burpees": 15,
    "lunges": 16,
    "lunges (complex)": 17,
    "bench-dips": 18,
}

LIMB_COLUMNS = {
    "left_wrist": ["left_arm_acc_x", "left_arm_acc_y", "left_arm_acc_z"],
    "right_wrist": ["right_arm_acc_x", "right_arm_acc_y", "right_arm_acc_z"],
    "left_ankle": ["left_leg_acc_x", "left_leg_acc_y", "left_leg_acc_z"],
    "right_ankle": ["right_leg_acc_x", "right_leg_acc_y", "right_leg_acc_z"],
}


def _interpolate_short_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    output = values.copy()
    for channel in range(output.shape[1]):
        signal = output[:, channel]
        missing = ~np.isfinite(signal)
        changes = np.diff(np.pad(missing.astype(np.int8), (1, 1)))
        starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        for start, stop in zip(starts, stops):
            if stop - start > max_gap or start == 0 or stop == len(signal):
                continue
            if np.isfinite(signal[start - 1]) and np.isfinite(signal[stop]):
                signal[start:stop] = np.linspace(
                    signal[start - 1], signal[stop], stop - start + 2, dtype=np.float32
                )[1:-1]
    return output


def repair_recording(
    arrays: dict[str, np.ndarray], strategy: str, max_gap: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Repair a full recording while preserving original validity masks."""
    validity = {key: np.isfinite(value) for key, value in arrays.items()}
    if strategy == "error" and not all(mask.all() for mask in validity.values()):
        fractions = {key: float(1.0 - mask.mean()) for key, mask in validity.items()}
        raise ValueError(f"Non-finite WEAR samples encountered: {fractions}")
    repaired = {
        key: _interpolate_short_gaps(value, max_gap) if strategy != "error" else value.copy()
        for key, value in arrays.items()
    }
    if strategy == "contralateral":
        for left, right in (("left_wrist", "right_wrist"), ("left_ankle", "right_ankle")):
            left_original, right_original = repaired[left].copy(), repaired[right].copy()
            left_missing = ~np.isfinite(repaired[left])
            right_missing = ~np.isfinite(repaired[right])
            repaired[left][left_missing & np.isfinite(right_original)] = right_original[
                left_missing & np.isfinite(right_original)
            ]
            repaired[right][right_missing & np.isfinite(left_original)] = left_original[
                right_missing & np.isfinite(left_original)
            ]
    repaired = {
        key: np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        for key, value in repaired.items()
    }
    return repaired, validity


def encode_labels(values: pd.Series) -> np.ndarray:
    """Map official activity strings to 1..18 and null/NaN to background class 0."""
    normalized = values.fillna("null").astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized.unique()).difference({"null", *ACTIVITY_TO_LABEL}))
    if unknown:
        raise ValueError(f"Unknown WEAR activity labels: {unknown}")
    return normalized.map({"null": 0, **ACTIVITY_TO_LABEL}).to_numpy(dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert official WEAR raw/inertial/50hz/sbj_*.csv files to RCBD arrays"
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="data/wear")
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--context-seconds", type=float, default=64.0)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument(
        "--missing-strategy",
        choices=["error", "interpolate", "contralateral"],
        default="interpolate",
    )
    parser.add_argument("--max-interpolation-gap-seconds", type=float, default=2.0)
    parser.add_argument(
        "--unlabeled", action="store_true", help="Prepare challenge test CSVs without labels"
    )
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    csv_paths = sorted(input_dir.glob("sbj_*.csv"), key=lambda path: int(path.stem.split("_")[-1]))
    if not csv_paths:
        raise FileNotFoundError(f"No sbj_*.csv files found in {input_dir}")
    output_dir = Path(args.output_dir).resolve()
    array_dir = output_dir / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    context = int(round(args.context_seconds * args.sample_rate))
    stride = int(round(args.stride_seconds * args.sample_rate))
    if context <= 0 or stride <= 0:
        raise ValueError("context and stride must be positive")
    max_gap = int(round(args.max_interpolation_gap_seconds * args.sample_rate))
    if max_gap < 0:
        raise ValueError("max interpolation gap must be non-negative")
    left_context = context // 2
    right_context = context - left_context
    rows: list[dict[str, object]] = []
    for csv_path in csv_paths:
        participant = csv_path.stem
        frame = pd.read_csv(csv_path, low_memory=False)
        required = {column for columns in LIMB_COLUMNS.values() for column in columns}
        if not args.unlabeled:
            required.add("label")
        if missing := required.difference(frame.columns):
            raise ValueError(f"{csv_path.name} lacks official WEAR columns: {sorted(missing)}")
        paths: dict[str, Path] = {}
        raw_arrays = {
            limb: frame[columns].to_numpy(dtype=np.float32)
            for limb, columns in LIMB_COLUMNS.items()
        }
        arrays, validity = repair_recording(raw_arrays, args.missing_strategy, max_gap)
        missing_fractions = {key: float(1.0 - mask.mean()) for key, mask in validity.items()}
        validity_paths: dict[str, Path] = {}
        for limb in LIMB_COLUMNS:
            values = arrays[limb]
            path = array_dir / f"{participant}_{limb}.npy"
            np.save(path, values, allow_pickle=False)
            paths[limb] = path
            validity_path = array_dir / f"{participant}_{limb}_validity.npy"
            np.save(validity_path, validity[limb].astype(np.uint8), allow_pickle=False)
            validity_paths[limb] = validity_path
        labels = (
            np.full(len(frame), -1, dtype=np.int64)
            if args.unlabeled
            else encode_labels(frame["label"])
        )
        labels_path = array_dir / f"{participant}_labels.npy"
        np.save(labels_path, labels, allow_pickle=False)
        for center in range(0, len(frame), stride):
            rows.append(
                {
                    "participant": participant,
                    "session": "official_csv",
                    "recording_id": participant,
                    "center": center,
                    "num_samples": len(frame),
                    "sample_rate": args.sample_rate,
                    "labels_path": str(labels_path),
                    "start": center - left_context,
                    "stop": center + right_context,
                    "label": int(labels[center]),
                    **{f"{limb}_path": str(path) for limb, path in paths.items()},
                    **{f"{limb}_validity_path": str(path) for limb, path in validity_paths.items()},
                }
            )
        print(
            f"converted {participant}: {len(frame)} samples, "
            f"{len(range(0, len(frame), stride))} windows, missing={missing_fractions}"
        )
    manifest = output_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False, encoding="utf-8")
    mapping = {"null": 0, **ACTIVITY_TO_LABEL}
    (output_dir / "label_mapping.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
