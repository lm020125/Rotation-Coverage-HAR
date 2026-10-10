from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SAMPLE_RATE = 50.0
RAW_COLUMNS = 120
SENSOR_WIDTH = 13
LABEL_COLUMN = 119

# Official order from the REALDISP dataset manual and the WIMUSim reader.
SENSOR_ORDER = ("RLA", "RUA", "BACK", "LUA", "LLA", "RC", "RT", "LT", "LC")
MODALITY_OFFSETS = {
    "acc": tuple(range(0, 3)),
    "acc_gyro": tuple(range(0, 6)),
    "acc_gyro_mag": tuple(range(0, 9)),
}
SENSOR_SETS = {
    # Closest match to WEAR's arm/leg placement: lower arms and calves.
    "distal": {
        "left_wrist": "LLA",
        "right_wrist": "RLA",
        "left_ankle": "LC",
        "right_ankle": "RC",
    },
    "proximal": {
        "left_wrist": "LUA",
        "right_wrist": "RUA",
        "left_ankle": "LT",
        "right_ankle": "RT",
    },
}
LIMBS = ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
SCENARIO_ORDER = {"ideal": 0, "self": 1, "mutual": 2}
FILE_PATTERN = re.compile(
    r"^subject(?P<subject>\d+)_(?P<session>ideal|self|mutual(?P<mutual_count>[4-7]))\.log$"
)

ACTIVITY_NAMES = (
    "Null",
    "Walking",
    "Jogging",
    "Running",
    "Jump up",
    "Jump front & back",
    "Jump sideways",
    "Jump leg/arms open/closed",
    "Jump rope",
    "Trunk twist (arms outstretched)",
    "Trunk twist (elbows bent)",
    "Waist bends forward",
    "Waist rotation",
    "Waist bends (reach foot with opposite hand)",
    "Reach heels backwards",
    "Lateral bend",
    "Lateral bend with arm up",
    "Repetitive forward stretching",
    "Upper trunk and lower body opposite twist",
    "Lateral elevation of arms",
    "Frontal elevation of arms",
    "Frontal hand claps",
    "Frontal crossing of arms",
    "Shoulders high-amplitude rotation",
    "Shoulders low-amplitude rotation",
    "Arms inner rotation",
    "Knees (alternating) to the breast",
    "Heels (alternating) to the backside",
    "Knees bending (crouching)",
    "Knees (alternating) bending forward",
    "Rotation on the knees",
    "Rowing",
    "Elliptical bike",
    "Cycling",
)


def parse_recording(path: Path) -> dict[str, object]:
    match = FILE_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected REALDISP filename: {path.name}")
    session = match.group("session")
    scenario = "mutual" if session.startswith("mutual") else session
    mutual_count = int(match.group("mutual_count")) if match.group("mutual_count") else None
    subject_id = int(match.group("subject"))
    return {
        "path": path,
        "subject_id": subject_id,
        "participant": f"subject{subject_id}",
        "session": session,
        "scenario": scenario,
        "mutual_displaced": mutual_count,
    }


def discover_recordings(
    input_dir: Path, scenarios: set[str], subjects: set[int] | None
) -> list[dict[str, object]]:
    recordings = [parse_recording(path) for path in input_dir.glob("subject*.log")]
    recordings = [
        item
        for item in recordings
        if item["scenario"] in scenarios
        and (subjects is None or int(item["subject_id"]) in subjects)
    ]
    recordings.sort(
        key=lambda item: (
            int(item["subject_id"]),
            SCENARIO_ORDER[str(item["scenario"])],
            str(item["session"]),
        )
    )
    if not recordings:
        raise FileNotFoundError(
            f"No REALDISP logs matching scenarios={sorted(scenarios)} in {input_dir}"
        )
    return recordings


def selected_raw_columns(
    sensor_set: str, modalities: str, include_anchor: bool = False
) -> dict[str, list[int]]:
    offsets = MODALITY_OFFSETS[modalities]
    result: dict[str, list[int]] = {}
    for limb, sensor in SENSOR_SETS[sensor_set].items():
        sensor_index = SENSOR_ORDER.index(sensor)
        start = 2 + SENSOR_WIDTH * sensor_index
        result[limb] = [start + offset for offset in offsets]
    if include_anchor:
        sensor_index = SENSOR_ORDER.index("BACK")
        start = 2 + SENSOR_WIDTH * sensor_index
        result["anchor"] = [start + offset for offset in offsets]
    return result


def first_row_width(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        line = handle.readline()
    return len(line.rstrip("\r\n").split("\t"))


def load_recording(
    path: Path, column_map: dict[str, list[int]], exclude_null: bool
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    width = first_row_width(path)
    if width != RAW_COLUMNS:
        raise ValueError(f"{path.name} has {width} columns; expected {RAW_COLUMNS}")
    sensor_keys = tuple(column_map)
    feature_columns = [column for sensor in sensor_keys for column in column_map[sensor]]
    usecols = feature_columns + [LABEL_COLUMN]
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=usecols,
        dtype=np.float32,
        engine="c",
    )
    # pandas may normalize ``usecols`` to source-file order. Reorder explicitly so
    # the matrix stays in LIMBS/column_map order before it is sliced below.
    frame = frame.loc[:, usecols]
    values = frame.to_numpy(dtype=np.float32, copy=False)
    channel_count = len(next(iter(column_map.values())))
    arrays: dict[str, np.ndarray] = {}
    validity: dict[str, np.ndarray] = {}
    cursor = 0
    for limb in sensor_keys:
        raw = np.ascontiguousarray(values[:, cursor : cursor + channel_count])
        mask = np.isfinite(raw)
        arrays[limb] = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        validity[limb] = mask.astype(np.uint8)
        cursor += channel_count
    raw_labels = values[:, -1]
    if not np.isfinite(raw_labels).all():
        raise ValueError(f"{path.name} contains non-finite activity labels")
    rounded = np.rint(raw_labels)
    if not np.allclose(raw_labels, rounded, rtol=0.0, atol=1e-5):
        raise ValueError(f"{path.name} contains non-integer activity labels")
    labels = rounded.astype(np.int64)
    if labels.size == 0 or labels.min() < 0 or labels.max() >= len(ACTIVITY_NAMES):
        raise ValueError(
            f"{path.name} labels outside 0..{len(ACTIVITY_NAMES) - 1}: "
            f"min={labels.min(initial=-1)}, max={labels.max(initial=-1)}"
        )
    if exclude_null:
        labels = labels - 1
    return arrays, validity, labels


def save_or_load_recording(
    record: dict[str, object],
    array_dir: Path,
    column_map: dict[str, list[int]],
    exclude_null: bool,
    overwrite: bool,
) -> tuple[dict[str, Path], dict[str, Path], Path, int, dict[str, float], np.ndarray]:
    prefix = f"subject{int(record['subject_id']):02d}_{record['session']}"
    sensor_keys = tuple(column_map)
    sensor_paths = {limb: array_dir / f"{prefix}_{limb}.npy" for limb in sensor_keys}
    validity_paths = {
        limb: array_dir / f"{prefix}_{limb}_validity.npy" for limb in sensor_keys
    }
    labels_path = array_dir / f"{prefix}_labels.npy"
    expected_paths = [*sensor_paths.values(), *validity_paths.values(), labels_path]
    if not overwrite and all(path.exists() for path in expected_paths):
        arrays = {
            limb: np.load(path, mmap_mode="r", allow_pickle=False)
            for limb, path in sensor_paths.items()
        }
        validity = {
            limb: np.load(path, mmap_mode="r", allow_pickle=False)
            for limb, path in validity_paths.items()
        }
        labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    else:
        arrays, validity, labels = load_recording(
            Path(record["path"]), column_map, exclude_null
        )
        for limb in sensor_keys:
            np.save(sensor_paths[limb], arrays[limb], allow_pickle=False)
            np.save(validity_paths[limb], validity[limb], allow_pickle=False)
        np.save(labels_path, labels, allow_pickle=False)
    lengths = {len(labels), *(len(arrays[limb]) for limb in sensor_keys)}
    if len(lengths) != 1:
        raise ValueError(f"Unaligned arrays for {prefix}: lengths={sorted(lengths)}")
    channel_counts = {
        arrays[limb].shape[1] for limb in sensor_keys if arrays[limb].ndim == 2
    }
    expected_channels = len(next(iter(column_map.values())))
    if channel_counts != {expected_channels}:
        raise ValueError(
            f"Unexpected channels for {prefix}: {channel_counts}, expected {expected_channels}"
        )
    missing = {
        limb: float(1.0 - np.asarray(validity[limb], dtype=np.float64).mean())
        for limb in sensor_keys
    }
    return sensor_paths, validity_paths, labels_path, len(labels), missing, labels


def build_manifest_rows(
    record: dict[str, object],
    sensor_paths: dict[str, Path],
    validity_paths: dict[str, Path],
    labels_path: Path,
    labels: np.ndarray,
    context_samples: int,
    stride_samples: int,
    sample_rate: float,
    exclude_null: bool,
) -> list[dict[str, object]]:
    left_context = context_samples // 2
    right_context = context_samples - left_context
    rows: list[dict[str, object]] = []
    for center in range(0, len(labels), stride_samples):
        label = int(labels[center])
        if exclude_null and label < 0:
            continue
        rows.append(
            {
                "participant": str(record["participant"]),
                "subject_id": int(record["subject_id"]),
                "session": str(record["session"]),
                "scenario": str(record["scenario"]),
                "mutual_displaced": record["mutual_displaced"],
                "recording_id": f"{record['participant']}__{record['session']}",
                "center": center,
                "num_samples": len(labels),
                "sample_rate": sample_rate,
                "labels_path": str(labels_path.resolve()),
                "start": center - left_context,
                "stop": center + right_context,
                "label": label,
                **{
                    f"{limb}_path": str(sensor_paths[limb].resolve())
                    for limb in sensor_paths
                },
                **{
                    f"{limb}_validity_path": str(validity_paths[limb].resolve())
                    for limb in validity_paths
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert official REALDISP subject*.log files into continuous arrays and "
            "participant-safe RCBD-HAR manifests."
        )
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="data/realdisp_distal")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=sorted(SCENARIO_ORDER),
        default=["ideal", "self"],
    )
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument("--sensor-set", choices=sorted(SENSOR_SETS), default="distal")
    parser.add_argument("--modalities", choices=sorted(MODALITY_OFFSETS), default="acc")
    parser.add_argument(
        "--include-anchor",
        action="store_true",
        help="Retain the central BACK sensor under the common key 'anchor'.",
    )
    parser.add_argument("--sample-rate", type=float, default=SAMPLE_RATE)
    parser.add_argument("--context-seconds", type=float, default=8.0)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument(
        "--exclude-null",
        action="store_true",
        help="Drop null-centered windows and remap activity labels 1..33 to 0..32.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-files", type=int, help="Only for a preprocessing smoke test")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    output_dir = Path(args.output_dir).resolve()
    array_dir = output_dir / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    context_samples = int(round(args.context_seconds * args.sample_rate))
    stride_samples = int(round(args.stride_seconds * args.sample_rate))
    if context_samples <= 0 or stride_samples <= 0:
        raise ValueError("context and stride must be positive")
    if args.sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Raw REALDISP is {SAMPLE_RATE:g} Hz; resampling is not implemented, "
            "so --sample-rate must remain 50."
        )
    subjects = set(args.subjects) if args.subjects else None
    recordings = discover_recordings(input_dir, set(args.scenarios), subjects)
    if args.max_files is not None:
        if args.max_files <= 0:
            raise ValueError("--max-files must be positive")
        recordings = recordings[: args.max_files]
    column_map = selected_raw_columns(
        args.sensor_set, args.modalities, include_anchor=args.include_anchor
    )

    all_rows: list[dict[str, object]] = []
    audit_records: list[dict[str, object]] = []
    raw_label_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for index, record in enumerate(recordings, start=1):
        sensor_paths, validity_paths, labels_path, samples, missing, labels = (
            save_or_load_recording(
                record,
                array_dir,
                column_map,
                args.exclude_null,
                args.overwrite,
            )
        )
        rows = build_manifest_rows(
            record,
            sensor_paths,
            validity_paths,
            labels_path,
            labels,
            context_samples,
            stride_samples,
            args.sample_rate,
            args.exclude_null,
        )
        all_rows.extend(rows)
        raw_label_counts[str(record["scenario"])].update(
            int(value) for value in np.asarray(labels) if int(value) >= 0
        )
        audit_records.append(
            {
                "file": Path(record["path"]).name,
                "participant": record["participant"],
                "scenario": record["scenario"],
                "session": record["session"],
                "samples": samples,
                "duration_seconds": samples / args.sample_rate,
                "windows": len(rows),
                "label_min": int(np.asarray(labels).min()),
                "label_max": int(np.asarray(labels).max()),
                "missing_fraction": missing,
            }
        )
        print(
            f"[{index}/{len(recordings)}] {Path(record['path']).name}: "
            f"{samples} samples, {len(rows)} windows, missing={missing}"
        )

    manifest = pd.DataFrame(all_rows).sort_values(
        ["subject_id", "scenario", "session", "center"], kind="stable"
    )
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8")
    for scenario in sorted(manifest["scenario"].unique(), key=SCENARIO_ORDER.get):
        manifest.loc[manifest["scenario"] == scenario].to_csv(
            output_dir / f"manifest_{scenario}.csv", index=False, encoding="utf-8"
        )

    label_mapping = (
        {name: index - 1 for index, name in enumerate(ACTIVITY_NAMES) if index > 0}
        if args.exclude_null
        else {name: index for index, name in enumerate(ACTIVITY_NAMES)}
    )
    (output_dir / "label_mapping.json").write_text(
        json.dumps(label_mapping, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "sensor_set": args.sensor_set,
        "sensor_mapping": {
            **SENSOR_SETS[args.sensor_set],
            **({"anchor": "BACK"} if args.include_anchor else {}),
        },
        "include_anchor": args.include_anchor,
        "modalities": args.modalities,
        "channels_per_sensor": len(next(iter(column_map.values()))),
        "sample_rate": args.sample_rate,
        "context_seconds": args.context_seconds,
        "stride_seconds": args.stride_seconds,
        "exclude_null": args.exclude_null,
        "num_classes": 33 if args.exclude_null else 34,
        "recordings": audit_records,
        "scenario_label_sample_counts": {
            scenario: {str(label): count for label, count in sorted(counter.items())}
            for scenario, counter in sorted(raw_label_counts.items())
        },
        "manifest_rows": len(manifest),
    }
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"wrote {len(manifest)} windows from {len(recordings)} recordings -> {manifest_path}"
    )


if __name__ == "__main__":
    main()
