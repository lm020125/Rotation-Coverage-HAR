from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_realdisp import (  # noqa: E402
    LABEL_COLUMN,
    RAW_COLUMNS,
    SENSOR_ORDER,
    SENSOR_WIDTH,
    load_recording,
    main as prepare_realdisp_main,
    parse_recording,
    selected_raw_columns,
)
from prepare_realdisp_wimusim import load_and_downsample  # noqa: E402


def write_fake_log(path: Path, samples: int = 100) -> np.ndarray:
    values = np.zeros((samples, RAW_COLUMNS), dtype=np.float32)
    values[:, 0] = np.arange(samples) // 50
    values[:, 1] = (np.arange(samples) % 50 + 1) * 20_000
    for sensor_index in range(len(SENSOR_ORDER)):
        start = 2 + SENSOR_WIDTH * sensor_index
        for offset in range(SENSOR_WIDTH):
            values[:, start + offset] = 100 * sensor_index + offset
    values[:, LABEL_COLUMN] = np.arange(samples) % 34
    np.savetxt(path, values, delimiter="\t", fmt="%.6f")
    return values


def test_realdisp_distal_column_mapping_matches_official_order(tmp_path: Path) -> None:
    path = tmp_path / "subject1_ideal.log"
    raw = write_fake_log(path, samples=8)
    column_map = selected_raw_columns("distal", "acc")
    arrays, validity, labels = load_recording(path, column_map, exclude_null=False)
    expected_sensors = {
        "left_wrist": "LLA",
        "right_wrist": "RLA",
        "left_ankle": "LC",
        "right_ankle": "RC",
    }
    for limb, sensor in expected_sensors.items():
        sensor_index = SENSOR_ORDER.index(sensor)
        start = 2 + SENSOR_WIDTH * sensor_index
        np.testing.assert_allclose(arrays[limb], raw[:, start : start + 3])
        assert validity[limb].all()
    np.testing.assert_array_equal(labels, np.arange(8) % 34)


def test_realdisp_anchor_maps_back_with_nine_axis_input(tmp_path: Path) -> None:
    path = tmp_path / "subject1_ideal.log"
    raw = write_fake_log(path, samples=8)
    column_map = selected_raw_columns("distal", "acc_gyro_mag", include_anchor=True)
    arrays, validity, _ = load_recording(path, column_map, exclude_null=False)
    sensor_index = SENSOR_ORDER.index("BACK")
    start = 2 + SENSOR_WIDTH * sensor_index
    np.testing.assert_allclose(arrays["anchor"], raw[:, start : start + 9])
    assert validity["anchor"].all()


def test_realdisp_parser_and_null_exclusion(tmp_path: Path) -> None:
    path = tmp_path / "subject15_mutual7.log"
    write_fake_log(path, samples=68)
    metadata = parse_recording(path)
    assert metadata["subject_id"] == 15
    assert metadata["scenario"] == "mutual"
    assert metadata["mutual_displaced"] == 7
    _, _, labels = load_recording(
        path, selected_raw_columns("distal", "acc"), exclude_null=True
    )
    assert labels[0] == -1
    assert labels[1] == 0
    assert labels.max() == 32


def test_realdisp_converter_writes_scenario_manifests(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir, output_dir = tmp_path / "raw", tmp_path / "processed"
    input_dir.mkdir()
    write_fake_log(input_dir / "subject1_ideal.log", samples=100)
    write_fake_log(input_dir / "subject1_self.log", samples=75)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_realdisp.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--scenarios",
            "ideal",
            "self",
            "--context-seconds",
            "1",
            "--stride-seconds",
            "0.5",
        ],
    )
    prepare_realdisp_main()
    manifest = pd.read_csv(output_dir / "manifest.csv")
    assert set(manifest["scenario"]) == {"ideal", "self"}
    assert len(pd.read_csv(output_dir / "manifest_ideal.csv")) == 4
    assert len(pd.read_csv(output_dir / "manifest_self.csv")) == 3
    sample = np.load(manifest.iloc[0].left_wrist_path, allow_pickle=False)
    assert sample.shape == (100, 3)


def test_wimusim_two_arm_six_axis_downsampling(tmp_path: Path) -> None:
    path = tmp_path / "subject1_ideal.log"
    raw = write_fake_log(path, samples=10)
    arrays, validity, labels = load_and_downsample(path)
    for side, sensor in {"left": "LLA", "right": "RLA"}.items():
        sensor_index = SENSOR_ORDER.index(sensor)
        start = 2 + SENSOR_WIDTH * sensor_index
        np.testing.assert_allclose(arrays[side], raw[::2, start : start + 6])
        assert validity[side].all()
    np.testing.assert_array_equal(labels, raw[::2, LABEL_COLUMN].astype(int) - 1)
