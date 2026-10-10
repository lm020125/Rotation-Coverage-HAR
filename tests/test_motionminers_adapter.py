from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_motionminers_mp1 import prepare_dataset  # noqa: E402
from prepare_motionminers_mp_b_loso import prepare_mp_b_loso  # noqa: E402


def _record(root: Path, subset: str, name: str, activity: int, misplacement: int) -> None:
    folder = root / subset / name
    folder.mkdir(parents=True)
    samples = 1000
    values = np.arange(samples, dtype=np.float32)
    frame = pd.DataFrame({"Time": values * 10.0, "Class": float(activity)})
    for prefix in ("", "Correct_"):
        if subset == "MP_A" and prefix:
            continue
        for side in ("L", "R"):
            for axis, offset in zip("XYZ", (1.0, 2.0, 3.0), strict=True):
                frame[f"{prefix}Acc{axis}_{side}"] = values + offset
    frame.to_csv(folder / f"{name}_data.csv", index=False)
    pd.DataFrame({"Class": np.full(samples, misplacement)}).to_csv(
        folder / f"{name}_misplacement_labels.csv", index=False
    )


def test_motionminers_protocol_is_subject_disjoint_and_paired(tmp_path) -> None:
    raw = tmp_path / "raw"
    _record(raw, "MP_A", "S01_R01", activity=3, misplacement=0)
    _record(raw, "MP_A", "S02_R01", activity=2, misplacement=0)
    _record(raw, "MP_A", "S03_R01", activity=3, misplacement=0)
    _record(raw, "MP_A", "S03_R02", activity=4, misplacement=0)
    _record(raw, "MP_B", "S04_R01", activity=3, misplacement=2)
    _record(raw, "MP_B", "S05_R01", activity=2, misplacement=5)
    _record(raw, "MP_B", "S06_R01", activity=3, misplacement=7)
    _record(raw, "MP_B", "S07_R01", activity=2, misplacement=9)

    output = tmp_path / "prepared"
    summary = prepare_dataset(raw, output, context_seconds=8.0, stride_seconds=0.5)
    source = pd.read_csv(output / "manifest_source_correct.csv")
    correct = pd.read_csv(output / "manifest_target_correct.csv")
    misplaced = pd.read_csv(output / "manifest_target_misplaced.csv")

    assert set(source.participant) == {"S01", "S02", "S03"}
    assert set(correct.participant) == {"S04", "S05", "S06", "S07"}
    assert not set(source.participant) & set(correct.participant)
    assert correct.pair_id.tolist() == misplaced.pair_id.tolist()
    assert correct.label.tolist() == misplaced.label.tolist()
    assert set(source.label) == {0, 1}
    assert summary["source_recordings"] == 3
    assert summary["target_recordings"] == 4


def test_motionminers_mp_b_loso_is_paired_and_stratified(tmp_path) -> None:
    raw = tmp_path / "raw"
    for subject in range(4, 8):
        _record(
            raw,
            "MP_B",
            f"S{subject:02d}_R01",
            activity=3,
            misplacement=6,
        )
        _record(
            raw,
            "MP_B",
            f"S{subject:02d}_R02",
            activity=2,
            misplacement=13,
        )

    output = tmp_path / "prepared_loso"
    summary = prepare_mp_b_loso(raw, output, context_seconds=8.0, stride_seconds=0.5)
    correct = pd.read_csv(output / "manifest_correct.csv")
    misplaced = pd.read_csv(output / "manifest_misplaced.csv")

    assert set(correct.participant) == {"S04", "S05", "S06", "S07"}
    assert correct.pair_id.tolist() == misplaced.pair_id.tolist()
    assert set(correct.label) == {0, 1}
    assert set(correct.misplacement_group) == {
        "unilateral_arm_rotation",
        "device_permutation",
    }
    assert correct.arm_input_affected.astype(bool).all()
    assert summary["recordings"] == 8
    assert summary["paired_windows"] == len(correct)
