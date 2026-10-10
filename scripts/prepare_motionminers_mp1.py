from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


RECORD_RE = re.compile(r"S(?P<subject>\d{2})_R(?P<recording>\d{2})$")
ACC_COLUMNS = {
    "left": ["AccX_L", "AccY_L", "AccZ_L"],
    "right": ["AccX_R", "AccY_R", "AccZ_R"],
}
CORRECT_ACC_COLUMNS = {
    side: [f"Correct_{column}" for column in columns]
    for side, columns in ACC_COLUMNS.items()
}
ACTIVITY_MAP = {2: 0, 3: 1}


def discover_recordings(root: Path, subset: str) -> list[Path]:
    subset_dir = root / subset
    if not subset_dir.is_dir():
        raise FileNotFoundError(f"MotionMiners subset is missing: {subset_dir}")
    recordings = [path for path in subset_dir.iterdir() if path.is_dir() and RECORD_RE.fullmatch(path.name)]
    return sorted(recordings, key=lambda path: tuple(map(int, RECORD_RE.fullmatch(path.name).groups())))


def _read_recording(recording_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    data_path = recording_dir / f"{recording_dir.name}_data.csv"
    misplacement_path = recording_dir / f"{recording_dir.name}_misplacement_labels.csv"
    if not data_path.exists() or not misplacement_path.exists():
        raise FileNotFoundError(f"Incomplete annotated recording: {recording_dir}")
    data = pd.read_csv(data_path)
    misplacement = pd.read_csv(misplacement_path)["Class"].to_numpy()
    if len(data) != len(misplacement):
        raise ValueError(f"Data/misplacement length mismatch in {recording_dir.name}")
    return data, np.rint(misplacement).astype(np.int64)


def _constant_integer(values: np.ndarray, name: str, record: str) -> int:
    rounded = np.rint(values).astype(np.int64)
    if not np.allclose(values, rounded, rtol=0.0, atol=1e-5):
        raise ValueError(f"Non-integer {name} in {record}")
    unique = np.unique(rounded)
    if len(unique) != 1:
        raise ValueError(f"Expected one {name} per recording in {record}, got {unique.tolist()}")
    return int(unique[0])


def _save_pair(
    frame: pd.DataFrame,
    columns: dict[str, list[str]],
    array_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for side in ("left", "right"):
        missing = set(columns[side]).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing {sorted(missing)} in {prefix}")
        values = frame[columns[side]].to_numpy(dtype=np.float32, copy=True)
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError(f"Invalid {side} acceleration array for {prefix}: {values.shape}")
        path = array_dir / f"{prefix}_{side}.npy"
        np.save(path, np.ascontiguousarray(values), allow_pickle=False)
        paths[side] = path.resolve()
    return paths


def _window_rows(
    *,
    participant: str,
    recording_id: str,
    subset: str,
    condition: str,
    activity: int,
    misplacement: int,
    paths: dict[str, Path],
    samples: int,
    context_samples: int,
    stride_samples: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for start in range(0, samples - context_samples + 1, stride_samples):
        stop = start + context_samples
        pair_id = f"{recording_id}:{start}:{stop}"
        rows.append(
            {
                "participant": participant,
                "recording_id": recording_id,
                "pair_id": pair_id,
                "subset": subset,
                "condition": condition,
                "source_activity_label": activity,
                "misplacement_class": misplacement,
                "sample_rate": 100,
                "start": start,
                "stop": stop,
                "center": start + context_samples // 2,
                "num_samples": samples,
                "label": ACTIVITY_MAP[activity],
                "left_path": str(paths["left"]),
                "right_path": str(paths["right"]),
            }
        )
    return rows


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    *,
    context_seconds: float = 8.0,
    stride_seconds: float = 0.5,
    max_source_files: int | None = None,
    max_target_files: int | None = None,
) -> dict[str, object]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    array_dir = output_dir / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    context_samples = int(round(context_seconds * 100.0))
    stride_samples = int(round(stride_seconds * 100.0))
    if context_samples <= 0 or stride_samples <= 0:
        raise ValueError("Window and stride must be positive")

    source_rows: list[dict[str, object]] = []
    target_rows = {"correct": [], "misplaced": []}
    audit: list[dict[str, object]] = []

    source_recordings = discover_recordings(input_dir, "MP_A")
    if max_source_files is not None:
        source_recordings = source_recordings[:max_source_files]
    for recording_dir in source_recordings:
        match = RECORD_RE.fullmatch(recording_dir.name)
        assert match is not None
        subject = int(match.group("subject"))
        if subject not in {1, 2, 3}:
            continue
        data, misplacement_values = _read_recording(recording_dir)
        activity = _constant_integer(data["Class"].to_numpy(), "activity", recording_dir.name)
        misplacement = _constant_integer(misplacement_values, "misplacement", recording_dir.name)
        if activity not in ACTIVITY_MAP or misplacement != 0:
            continue
        paths = _save_pair(data, ACC_COLUMNS, array_dir, f"source_{recording_dir.name}")
        rows = _window_rows(
            participant=f"S{subject:02d}", recording_id=recording_dir.name, subset="MP_A",
            condition="correct", activity=activity, misplacement=misplacement, paths=paths,
            samples=len(data), context_samples=context_samples, stride_samples=stride_samples,
        )
        source_rows.extend(rows)
        audit.append({"recording": recording_dir.name, "role": "source", "windows": len(rows)})

    target_recordings = discover_recordings(input_dir, "MP_B")
    if max_target_files is not None:
        target_recordings = target_recordings[:max_target_files]
    for recording_dir in target_recordings:
        match = RECORD_RE.fullmatch(recording_dir.name)
        assert match is not None
        subject = int(match.group("subject"))
        if subject not in {4, 5, 6, 7}:
            continue
        data, misplacement_values = _read_recording(recording_dir)
        activity = _constant_integer(data["Class"].to_numpy(), "activity", recording_dir.name)
        misplacement = _constant_integer(misplacement_values, "misplacement", recording_dir.name)
        if activity not in ACTIVITY_MAP or misplacement == 0:
            continue
        misplaced_paths = _save_pair(
            data, ACC_COLUMNS, array_dir, f"target_misplaced_{recording_dir.name}"
        )
        correct_paths = _save_pair(
            data, CORRECT_ACC_COLUMNS, array_dir, f"target_correct_{recording_dir.name}"
        )
        common = dict(
            participant=f"S{subject:02d}", recording_id=recording_dir.name, subset="MP_B",
            activity=activity, misplacement=misplacement, samples=len(data),
            context_samples=context_samples, stride_samples=stride_samples,
        )
        misplaced = _window_rows(condition="misplaced", paths=misplaced_paths, **common)
        correct = _window_rows(condition="correct", paths=correct_paths, **common)
        if [row["pair_id"] for row in correct] != [row["pair_id"] for row in misplaced]:
            raise RuntimeError(f"Paired target windows diverged for {recording_dir.name}")
        target_rows["correct"].extend(correct)
        target_rows["misplaced"].extend(misplaced)
        audit.append({"recording": recording_dir.name, "role": "target_pair", "windows": len(correct)})

    if not source_rows or not target_rows["correct"]:
        raise ValueError("The strict MP1 protocol produced an empty source or target split")
    source = pd.DataFrame(source_rows).sort_values(["participant", "recording_id", "start"])
    correct = pd.DataFrame(target_rows["correct"]).sort_values(["participant", "recording_id", "start"])
    misplaced = pd.DataFrame(target_rows["misplaced"]).sort_values(["participant", "recording_id", "start"])
    if source.participant.isin(correct.participant).any():
        raise RuntimeError("Source and target participants overlap")
    if correct["pair_id"].tolist() != misplaced["pair_id"].tolist():
        raise RuntimeError("Correct and misplaced MP_B manifests are not exactly paired")
    output_dir.mkdir(parents=True, exist_ok=True)
    source.to_csv(output_dir / "manifest_source_correct.csv", index=False)
    correct.to_csv(output_dir / "manifest_target_correct.csv", index=False)
    misplaced.to_csv(output_dir / "manifest_target_misplaced.csv", index=False)
    summary: dict[str, object] = {
        "source": "MotionMiners MP1 public dataset (Zenodo 10.5281/zenodo.8272091)",
        "task": "binary activity recognition under unseen-participant sensor misplacement",
        "source_participants": sorted(source.participant.unique().tolist()),
        "target_participants": sorted(correct.participant.unique().tolist()),
        "activity_mapping": {str(key): value for key, value in ACTIVITY_MAP.items()},
        "modalities": ["left-arm acceleration", "right-arm acceleration"],
        "sample_rate": 100,
        "context_seconds": context_seconds,
        "stride_seconds": stride_seconds,
        "source_windows": len(source),
        "target_paired_windows": len(correct),
        "source_recordings": int(source.recording_id.nunique()),
        "target_recordings": int(correct.recording_id.nunique()),
        "protocol_guards": {
            "source_target_subject_disjoint": True,
            "target_correct_misplaced_exactly_paired": True,
            "only_full_windows": True,
            "target_not_used_for_selection": True,
        },
        "recordings": audit,
    }
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the strict MotionMiners MP1 external protocol")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-seconds", type=float, default=8.0)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--max-source-files", type=int)
    parser.add_argument("--max-target-files", type=int)
    args = parser.parse_args()
    summary = prepare_dataset(
        Path(args.input_dir), Path(args.output_dir), context_seconds=args.context_seconds,
        stride_seconds=args.stride_seconds, max_source_files=args.max_source_files,
        max_target_files=args.max_target_files,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
