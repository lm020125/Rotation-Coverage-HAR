from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_realdisp import (
    ACTIVITY_NAMES,
    LABEL_COLUMN,
    RAW_COLUMNS,
    SAMPLE_RATE,
    SENSOR_ORDER,
    SENSOR_WIDTH,
    discover_recordings,
    first_row_width,
)


TARGET_RATE = 25.0
SENSORS = {"left": "LLA", "right": "RLA"}
OFFSETS = tuple(range(6))  # Acc XYZ followed by Gyro XYZ.


def raw_columns() -> dict[str, list[int]]:
    output: dict[str, list[int]] = {}
    for side, sensor in SENSORS.items():
        start = 2 + SENSOR_WIDTH * SENSOR_ORDER.index(sensor)
        output[side] = [start + offset for offset in OFFSETS]
    return output


def load_and_downsample(
    path: Path, factor: int = 2
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    if first_row_width(path) != RAW_COLUMNS:
        raise ValueError(f"{path.name} does not contain {RAW_COLUMNS} tab-separated columns")
    columns = raw_columns()
    feature_columns = [column for side in ("left", "right") for column in columns[side]]
    usecols = feature_columns + [LABEL_COLUMN]
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=usecols,
        dtype=np.float32,
        engine="c",
    ).loc[:, usecols]
    values = frame.to_numpy(dtype=np.float32, copy=False)[::factor]
    arrays: dict[str, np.ndarray] = {}
    validity: dict[str, np.ndarray] = {}
    for index, side in enumerate(("left", "right")):
        raw = np.ascontiguousarray(values[:, index * 6 : (index + 1) * 6])
        mask = np.isfinite(raw)
        arrays[side] = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        validity[side] = mask.astype(np.uint8)
    raw_labels = values[:, -1]
    if not np.isfinite(raw_labels).all():
        raise ValueError(f"{path.name} contains non-finite labels")
    labels = np.rint(raw_labels).astype(np.int64)
    if not np.allclose(raw_labels, labels, rtol=0.0, atol=1e-5):
        raise ValueError(f"{path.name} contains non-integer labels")
    if labels.size == 0 or labels.min() < 0 or labels.max() >= len(ACTIVITY_NAMES):
        raise ValueError(f"{path.name} contains labels outside 0..33")
    return arrays, validity, labels - 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the public WIMUSim REALDISP protocol: RLA+LLA Acc/Gyro, "
            "50-to-25 Hz decimation, four-second windows and one-second stride."
        )
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenarios", nargs="+", default=["ideal", "self"])
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument("--max-files", type=int, help="Only for a real-data smoke test")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    array_dir = output_dir / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    recordings = discover_recordings(
        input_dir,
        set(args.scenarios),
        subjects=set(args.subjects) if args.subjects else None,
    )
    if args.max_files is not None:
        if args.max_files <= 0:
            raise ValueError("--max-files must be positive")
        recordings = recordings[: args.max_files]
    context_samples = int(4.0 * TARGET_RATE)
    stride_samples = int(1.0 * TARGET_RATE)
    rows: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    label_counts: dict[str, Counter[int]] = defaultdict(Counter)

    for number, record in enumerate(recordings, start=1):
        prefix = f"subject{int(record['subject_id']):02d}_{record['session']}"
        paths = {side: array_dir / f"{prefix}_{side}.npy" for side in ("left", "right")}
        validity_paths = {
            side: array_dir / f"{prefix}_{side}_validity.npy" for side in ("left", "right")
        }
        labels_path = array_dir / f"{prefix}_labels.npy"
        expected = [*paths.values(), *validity_paths.values(), labels_path]
        if not args.overwrite and all(path.exists() for path in expected):
            arrays = {
                side: np.load(paths[side], mmap_mode="r", allow_pickle=False)
                for side in ("left", "right")
            }
            validity = {
                side: np.load(validity_paths[side], mmap_mode="r", allow_pickle=False)
                for side in ("left", "right")
            }
            labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
        else:
            arrays, validity, labels = load_and_downsample(Path(record["path"]))
            for side in ("left", "right"):
                np.save(paths[side], arrays[side], allow_pickle=False)
                np.save(validity_paths[side], validity[side], allow_pickle=False)
            np.save(labels_path, labels, allow_pickle=False)
        lengths = {len(labels), len(arrays["left"]), len(arrays["right"])}
        if len(lengths) != 1:
            raise ValueError(f"Unaligned downsampled arrays for {prefix}: {lengths}")
        recording_rows = 0
        for center in range(0, len(labels), stride_samples):
            label = int(labels[center])
            if label < 0:
                continue
            rows.append(
                {
                    "participant": str(record["participant"]),
                    "subject_id": int(record["subject_id"]),
                    "session": str(record["session"]),
                    "scenario": str(record["scenario"]),
                    "recording_id": f"{record['participant']}__{record['session']}",
                    "center": center,
                    "num_samples": len(labels),
                    "sample_rate": TARGET_RATE,
                    "labels_path": str(labels_path.resolve()),
                    "start": center - context_samples // 2,
                    "stop": center + context_samples // 2,
                    "label": label,
                    "left_path": str(paths["left"].resolve()),
                    "right_path": str(paths["right"].resolve()),
                    "left_validity_path": str(validity_paths["left"].resolve()),
                    "right_validity_path": str(validity_paths["right"].resolve()),
                }
            )
            recording_rows += 1
        label_counts[str(record["scenario"])].update(
            int(value) for value in np.asarray(labels) if int(value) >= 0
        )
        missing = {
            side: float(1.0 - np.asarray(validity[side], dtype=np.float64).mean())
            for side in ("left", "right")
        }
        audit.append(
            {
                "file": Path(record["path"]).name,
                "samples_25hz": len(labels),
                "windows": recording_rows,
                "missing_fraction": missing,
            }
        )
        print(
            f"[{number}/{len(recordings)}] {Path(record['path']).name}: "
            f"{len(labels)} samples, {recording_rows} windows"
        )

    manifest = pd.DataFrame(rows).sort_values(
        ["subject_id", "scenario", "session", "center"], kind="stable"
    )
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    for scenario in sorted(manifest.scenario.unique()):
        manifest.loc[manifest.scenario == scenario].to_csv(
            output_dir / f"manifest_{scenario}.csv", index=False
        )
    summary = {
        "source": "UCI REALDISP raw logs",
        "protocol": "WIMUSim REALDISP public paper",
        "sensors": SENSORS,
        "modalities": ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"],
        "raw_sample_rate": SAMPLE_RATE,
        "target_sample_rate": TARGET_RATE,
        "downsampling": "phase-locked factor-2 decimation",
        "context_seconds": 4.0,
        "stride_seconds": 1.0,
        "num_classes": 33,
        "raw_null_label_excluded": True,
        "manifest_rows": len(manifest),
        "recordings": audit,
        "scenario_label_sample_counts": {
            scenario: {str(label): count for label, count in sorted(counter.items())}
            for scenario, counter in sorted(label_counts.items())
        },
    }
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(manifest)} windows -> {output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
