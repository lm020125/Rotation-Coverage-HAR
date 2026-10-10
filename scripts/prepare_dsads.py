from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


SAMPLE_RATE = 25.0
SEGMENT_SAMPLES = 125
RAW_COLUMNS = 45
LIMBS = ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
SENSOR_BLOCKS = {
    "torso": 0,
    "right_wrist": 9,
    "left_wrist": 18,
    "right_ankle": 27,
    "left_ankle": 36,
}
MODALITY_WIDTHS = {"acc": 3, "acc_gyro": 6, "acc_gyro_mag": 9}
ACTIVITY_NAMES = (
    "sitting",
    "standing",
    "lying_on_back",
    "lying_on_right_side",
    "ascending_stairs",
    "descending_stairs",
    "standing_in_elevator",
    "moving_in_elevator",
    "walking_in_parking_lot",
    "walking_treadmill_flat_4kmh",
    "walking_treadmill_incline_4kmh",
    "running_treadmill_8kmh",
    "stepper",
    "cross_trainer",
    "cycling_horizontal",
    "cycling_vertical",
    "rowing",
    "jumping",
    "basketball",
)
SEGMENT_PATTERN = re.compile(r"^s(?P<segment>\d{2})\.txt$")
ACTIVITY_PATTERN = re.compile(r"^a(?P<activity>\d{2})$")
PARTICIPANT_PATTERN = re.compile(r"^p(?P<participant>\d+)$")


def parse_segment(path: Path) -> dict[str, int | str | Path]:
    segment_match = SEGMENT_PATTERN.match(path.name)
    participant_match = PARTICIPANT_PATTERN.match(path.parent.name)
    activity_match = ACTIVITY_PATTERN.match(path.parent.parent.name)
    if segment_match is None or participant_match is None or activity_match is None:
        raise ValueError(f"Unexpected DSADS path: {path}")
    activity_id = int(activity_match.group("activity"))
    participant_id = int(participant_match.group("participant"))
    segment_id = int(segment_match.group("segment"))
    if not 1 <= activity_id <= len(ACTIVITY_NAMES):
        raise ValueError(f"Activity outside 1..{len(ACTIVITY_NAMES)}: {path}")
    if not 1 <= participant_id <= 8 or not 1 <= segment_id <= 60:
        raise ValueError(f"Participant/segment outside the official DSADS range: {path}")
    return {
        "path": path,
        "activity_id": activity_id,
        "participant_id": participant_id,
        "participant": f"p{participant_id}",
        "segment_id": segment_id,
        "label": activity_id - 1,
    }


def discover_segments(input_dir: Path) -> list[dict[str, int | str | Path]]:
    records = [parse_segment(path) for path in input_dir.glob("a*/p*/s*.txt")]
    records.sort(
        key=lambda item: (
            int(item["activity_id"]),
            int(item["participant_id"]),
            int(item["segment_id"]),
        )
    )
    if not records:
        raise FileNotFoundError(f"No DSADS a*/p*/s*.txt files below {input_dir}")
    return records


def load_segment(
    path: Path, modalities: str, include_anchor: bool = False
) -> dict[str, np.ndarray]:
    width = MODALITY_WIDTHS[modalities]
    values = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if values.shape != (SEGMENT_SAMPLES, RAW_COLUMNS):
        raise ValueError(
            f"{path} has shape {values.shape}; expected {(SEGMENT_SAMPLES, RAW_COLUMNS)}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite sensor values")
    arrays = {
        limb: np.ascontiguousarray(values[:, start : start + width])
        for limb, start in SENSOR_BLOCKS.items()
        if limb in LIMBS
    }
    if include_anchor:
        arrays["anchor"] = np.ascontiguousarray(values[:, :width])
    return arrays


def validate_official_inventory(records: list[dict[str, int | str | Path]]) -> None:
    expected = len(ACTIVITY_NAMES) * 8 * 60
    if len(records) != expected:
        raise ValueError(f"Expected {expected} DSADS segments, found {len(records)}")
    counts = Counter(
        (int(item["activity_id"]), int(item["participant_id"])) for item in records
    )
    invalid = {key: value for key, value in counts.items() if value != 60}
    if len(counts) != len(ACTIVITY_NAMES) * 8 or invalid:
        raise ValueError(f"Incomplete activity-participant inventory: {invalid}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert official UCI DSADS text segments to four-limb RCBD arrays, "
            "optionally retaining the torso as a central anchor."
        )
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="data/dsads_accgyro")
    parser.add_argument(
        "--modalities", choices=sorted(MODALITY_WIDTHS), default="acc_gyro"
    )
    parser.add_argument(
        "--include-anchor",
        action="store_true",
        help="Retain the torso sensor under the common key 'anchor'.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-files", type=int, help="Only for preprocessing smoke tests")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    output_dir = Path(args.output_dir).resolve()
    array_dir = output_dir / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)

    records = discover_segments(input_dir)
    if args.max_files is None:
        validate_official_inventory(records)
    else:
        if args.max_files <= 0:
            raise ValueError("--max-files must be positive")
        records = records[: args.max_files]

    rows: list[dict[str, object]] = []
    sensor_keys = (*LIMBS, *(("anchor",) if args.include_anchor else ()))
    for index, record in enumerate(records, start=1):
        destination = array_dir / (
            f"a{int(record['activity_id']):02d}_p{int(record['participant_id'])}_"
            f"s{int(record['segment_id']):02d}.npz"
        )
        if args.overwrite or not destination.exists():
            arrays = load_segment(
                Path(record["path"]), args.modalities, include_anchor=args.include_anchor
            )
            np.savez(destination, **arrays)
        else:
            with np.load(destination, allow_pickle=False) as saved:
                missing = set(sensor_keys).difference(saved.files)
                if missing:
                    raise KeyError(
                        f"Existing processed segment lacks {sorted(missing)}: {destination}"
                    )
                arrays = {sensor: saved[sensor] for sensor in sensor_keys}
            expected_shape = (SEGMENT_SAMPLES, MODALITY_WIDTHS[args.modalities])
            if any(array.shape != expected_shape for array in arrays.values()):
                raise ValueError(f"Existing processed segment has wrong shape: {destination}")
        rows.append(
            {
                "path": str(destination.resolve()),
                "participant": str(record["participant"]),
                "subject_id": int(record["participant_id"]),
                "activity_id": int(record["activity_id"]),
                "activity": ACTIVITY_NAMES[int(record["label"])],
                "segment_id": int(record["segment_id"]),
                "sample_rate": SAMPLE_RATE,
                "num_samples": SEGMENT_SAMPLES,
                "label": int(record["label"]),
            }
        )
        if index % 1000 == 0 or index == len(records):
            print(f"[{index}/{len(records)}] converted DSADS segments")

    manifest = pd.DataFrame(rows).sort_values(
        ["subject_id", "activity_id", "segment_id"], kind="stable"
    )
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8")
    label_mapping = {name: index for index, name in enumerate(ACTIVITY_NAMES)}
    (output_dir / "label_mapping.json").write_text(
        json.dumps(label_mapping, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "source": "UCI Daily and Sports Activities Dataset, id=256",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "modalities": args.modalities,
        "include_anchor": args.include_anchor,
        "channels_per_limb": MODALITY_WIDTHS[args.modalities],
        "sensor_mapping": {
            "left_wrist": "left arm (LA), source columns 19-27",
            "right_wrist": "right arm (RA), source columns 10-18",
            "left_ankle": "left leg (LL), source columns 37-45",
            "right_ankle": "right leg (RL), source columns 28-36",
            **(
                {"anchor": "torso (T), source columns 1-9"}
                if args.include_anchor
                else {}
            ),
        },
        "sample_rate": SAMPLE_RATE,
        "samples_per_segment": SEGMENT_SAMPLES,
        "num_classes": len(ACTIVITY_NAMES),
        "manifest_rows": len(manifest),
        "participant_counts": {
            str(key): int(value)
            for key, value in manifest["participant"].value_counts().sort_index().items()
        },
        "label_counts": {
            str(key): int(value)
            for key, value in manifest["label"].value_counts().sort_index().items()
        },
    }
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {len(manifest)} segments -> {manifest_path}")


if __name__ == "__main__":
    main()
