from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_motionminers_mp1 import (  # noqa: E402
    ACC_COLUMNS,
    CORRECT_ACC_COLUMNS,
    RECORD_RE,
    _constant_integer,
    _read_recording,
    _save_pair,
    _window_rows,
    discover_recordings,
)


MP_B_PARTICIPANTS = ("S04", "S05", "S06", "S07")
ACTIVITY_LABELS = (2, 3)
MISPLACEMENT_GROUPS = {
    1: "device_permutation",
    2: "torso_only",
    3: "torso_only",
    4: "torso_only",
    5: "torso_only",
    6: "unilateral_arm_rotation",
    7: "unilateral_arm_rotation",
    8: "unilateral_arm_rotation",
    9: "unilateral_arm_rotation",
    10: "bilateral_arm_rotation",
    11: "torso_only",
    12: "torso_only",
    13: "device_permutation",
    14: "device_permutation",
    15: "torso_only",
}
ARM_AFFECTING_GROUPS = {
    "device_permutation",
    "unilateral_arm_rotation",
    "bilateral_arm_rotation",
}


def _annotate(rows: list[dict[str, object]], misplacement: int) -> None:
    group = MISPLACEMENT_GROUPS.get(misplacement, "unknown")
    for row in rows:
        row["misplacement_group"] = group
        row["arm_input_affected"] = group in ARM_AFFECTING_GROUPS
        row["session"] = row["recording_id"]


def prepare_mp_b_loso(
    input_dir: Path,
    output_dir: Path,
    *,
    context_seconds: float = 8.0,
    stride_seconds: float = 0.5,
    max_files: int | None = None,
) -> dict[str, object]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    array_dir = output_dir / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    context_samples = int(round(context_seconds * 100.0))
    stride_samples = int(round(stride_seconds * 100.0))
    if context_samples <= 0 or stride_samples <= 0:
        raise ValueError("Window and stride must be positive")

    correct_rows: list[dict[str, object]] = []
    misplaced_rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    recordings = discover_recordings(input_dir, "MP_B")
    if max_files is not None:
        recordings = recordings[:max_files]
    for recording_dir in recordings:
        match = RECORD_RE.fullmatch(recording_dir.name)
        assert match is not None
        participant = f"S{int(match.group('subject')):02d}"
        if participant not in MP_B_PARTICIPANTS:
            continue
        data, misplacement_values = _read_recording(recording_dir)
        activity = _constant_integer(data["Class"].to_numpy(), "activity", recording_dir.name)
        misplacement = _constant_integer(misplacement_values, "misplacement", recording_dir.name)
        if activity not in ACTIVITY_LABELS or misplacement == 0:
            continue
        if misplacement not in MISPLACEMENT_GROUPS:
            raise ValueError(f"Unknown MP_B misplacement {misplacement} in {recording_dir.name}")
        misplaced_paths = _save_pair(
            data, ACC_COLUMNS, array_dir, f"misplaced_{recording_dir.name}"
        )
        correct_paths = _save_pair(
            data, CORRECT_ACC_COLUMNS, array_dir, f"correct_{recording_dir.name}"
        )
        common = dict(
            participant=participant,
            recording_id=recording_dir.name,
            subset="MP_B",
            activity=activity,
            misplacement=misplacement,
            samples=len(data),
            context_samples=context_samples,
            stride_samples=stride_samples,
        )
        correct = _window_rows(condition="correct", paths=correct_paths, **common)
        misplaced = _window_rows(condition="misplaced", paths=misplaced_paths, **common)
        _annotate(correct, misplacement)
        _annotate(misplaced, misplacement)
        if [row["pair_id"] for row in correct] != [row["pair_id"] for row in misplaced]:
            raise RuntimeError(f"Paired windows diverged for {recording_dir.name}")
        correct_rows.extend(correct)
        misplaced_rows.extend(misplaced)
        records.append(
            {
                "participant": participant,
                "recording_id": recording_dir.name,
                "activity": activity,
                "label": correct[0]["label"] if correct else None,
                "misplacement_class": misplacement,
                "misplacement_group": MISPLACEMENT_GROUPS[misplacement],
                "samples": len(data),
                "windows": len(correct),
            }
        )

    if not correct_rows:
        raise ValueError("MP_B preparation produced no paired windows")
    sort_columns = ["participant", "recording_id", "start"]
    correct_frame = pd.DataFrame(correct_rows).sort_values(sort_columns).reset_index(drop=True)
    misplaced_frame = pd.DataFrame(misplaced_rows).sort_values(sort_columns).reset_index(drop=True)
    participants = tuple(sorted(correct_frame.participant.unique().tolist()))
    if participants != MP_B_PARTICIPANTS:
        raise ValueError(f"Expected MP_B participants {MP_B_PARTICIPANTS}, got {participants}")
    if correct_frame.pair_id.tolist() != misplaced_frame.pair_id.tolist():
        raise RuntimeError("Correct and misplaced MP_B manifests are not exactly paired")
    if set(correct_frame.label.unique()) != {0, 1}:
        raise ValueError("The frozen MP_B HAR task must contain both activity labels")
    if not bool(correct_frame.groupby("participant")["label"].nunique().eq(2).all()):
        raise ValueError("Every MP_B participant must contain both HAR classes")

    output_dir.mkdir(parents=True, exist_ok=True)
    correct_frame.to_csv(output_dir / "manifest_correct.csv", index=False)
    misplaced_frame.to_csv(output_dir / "manifest_misplaced.csv", index=False)
    pd.DataFrame(records).to_csv(output_dir / "record_audit.csv", index=False)
    summary: dict[str, object] = {
        "source": "MotionMiners MP1 MP_B (Zenodo 10.5281/zenodo.8272091)",
        "task": "paired binary HAR under participant-independent real misplacement",
        "participants": list(participants),
        "activity_mapping": {"2": 0, "3": 1},
        "sample_rate": 100,
        "context_seconds": context_seconds,
        "stride_seconds": stride_seconds,
        "paired_windows": len(correct_frame),
        "recordings": int(correct_frame.recording_id.nunique()),
        "recordings_per_participant": dict(Counter(row["participant"] for row in records)),
        "recordings_per_group": dict(Counter(row["misplacement_group"] for row in records)),
        "protocol_guards": {
            "correct_misplaced_exactly_paired": True,
            "all_outer_subjects_have_both_classes": True,
            "synchronisation_excluded": True,
            "torso_only_flagged_for_control_analysis": True,
            "no_smoothing_primary": True,
        },
    }
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare paired MotionMiners MP_B LOSO data")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-seconds", type=float, default=8.0)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--max-files", type=int)
    args = parser.parse_args()
    summary = prepare_mp_b_loso(
        Path(args.input_dir),
        Path(args.output_dir),
        context_seconds=args.context_seconds,
        stride_seconds=args.stride_seconds,
        max_files=args.max_files,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
