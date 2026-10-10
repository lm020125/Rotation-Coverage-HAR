from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_wear_official_csv import (  # noqa: E402
    LIMB_COLUMNS,
    encode_labels,
    main as prepare_official_main,
    repair_recording,
)
from rcbd_har.data import WindowManifestDataset, load_manifest  # noqa: E402


def test_official_wear_label_mapping_includes_background() -> None:
    labels = encode_labels(pd.Series([np.nan, "null", "jogging", "bench-dips"]))
    np.testing.assert_array_equal(labels, [0, 0, 1, 18])


def test_recording_repair_interpolates_only_short_gaps() -> None:
    arrays = {
        key: np.asarray(
            [[1.0, 1.0, 1.0], [np.nan] * 3, [3.0] * 3, [np.nan] * 3, [np.nan] * 3, [6.0] * 3]
        )
        for key in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
    }
    repaired, validity = repair_recording(arrays, "interpolate", max_gap=1)
    np.testing.assert_allclose(repaired["left_wrist"][1], [2.0] * 3)
    np.testing.assert_allclose(repaired["left_wrist"][3:5], 0.0)
    assert not validity["left_wrist"][1].any()


def test_official_csv_converter_writes_train_and_validity_arrays(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    input_dir.mkdir()
    data = {"label": ["jogging"] * 8}
    for columns in LIMB_COLUMNS.values():
        for axis, column in enumerate(columns):
            data[column] = np.arange(8, dtype=np.float32) + axis
    data["left_arm_acc_x"][3] = np.nan
    pd.DataFrame(data).to_csv(input_dir / "sbj_0.csv", index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_wear_official_csv.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--sample-rate",
            "1",
            "--context-seconds",
            "4",
            "--stride-seconds",
            "2",
        ],
    )
    prepare_official_main()
    frame = load_manifest(output_dir / "manifest.csv")
    assert {"recording_id", "num_samples", "left_wrist_validity_path"}.issubset(frame.columns)
    sample = WindowManifestDataset(
        frame, limb_mode="four", normalization="none", missing_strategy="interpolate"
    )[2]
    assert torch.isfinite(sample["left_wrist"]).all()
    assert sample["missing_fraction"][0] > 0
