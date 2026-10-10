from pathlib import Path

import numpy as np
import pandas as pd

from rcbd_har.data.manifest import (
    WindowManifestDataset,
    build_participant_split,
    estimate_global_normalization,
    load_manifest,
)
from rcbd_har.data.synthetic import create_four_limb_synthetic_dataset, create_synthetic_dataset


def test_synthetic_manifest_and_participant_split(tmp_path: Path) -> None:
    manifest = create_synthetic_dataset(tmp_path, participants=4, windows_per_participant=4)
    frame = load_manifest(manifest)
    split = build_participant_split(frame, "P01", "P02")
    train_people = set(frame.iloc[split.train_indices].participant)
    val_people = set(frame.iloc[split.val_indices].participant)
    test_people = set(frame.iloc[split.test_indices].participant)
    assert train_people.isdisjoint(val_people | test_people)
    assert val_people == {"P02"}
    assert test_people == {"P01"}


def test_dataset_normalizes_and_loads_equal_wrist_shapes(tmp_path: Path) -> None:
    manifest = create_synthetic_dataset(tmp_path, participants=3, windows_per_participant=2)
    dataset = WindowManifestDataset(manifest)
    sample = dataset[0]
    assert sample["left"].shape == sample["right"].shape
    np.testing.assert_allclose(sample["left"].numpy().mean(axis=0), 0.0, atol=1e-5)


def test_manifest_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"path": ["x.npz"]}).to_csv(path, index=False)
    try:
        load_manifest(path)
    except ValueError as error:
        assert "lacks required columns" in str(error)
    else:
        raise AssertionError("invalid manifest was accepted")


def test_four_limb_npz_dataset(tmp_path: Path) -> None:
    manifest = create_four_limb_synthetic_dataset(
        tmp_path, participants=3, windows_per_participant=2, length=40
    )
    sample = WindowManifestDataset(manifest, limb_mode="four")[0]
    for key in ("left_wrist", "right_wrist", "left_ankle", "right_ankle"):
        assert sample[key].shape == (40, 3)
        np.testing.assert_allclose(sample[key].numpy().mean(axis=0), 0.0, atol=1e-5)


def test_four_limb_npz_dataset_loads_optional_anchor(tmp_path: Path) -> None:
    path = tmp_path / "anchored.npz"
    arrays = {
        key: np.full((12, 9), index, dtype=np.float32)
        for index, key in enumerate(
            ("left_wrist", "right_wrist", "left_ankle", "right_ankle", "anchor"),
            start=1,
        )
    }
    np.savez(path, **arrays)
    frame = pd.DataFrame([{"path": str(path), "participant": "P01", "label": 0}])
    stats = estimate_global_normalization(frame, "four", mode="train_global")
    assert "anchor" in stats
    sample = WindowManifestDataset(
        frame,
        limb_mode="four",
        normalization="train_global",
        normalization_stats=stats,
    )[0]
    assert sample["anchor"].shape == (12, 9)
    assert sample["missing_fraction"].shape == (5,)


def test_four_limb_continuous_dataset_pads_edges(tmp_path: Path) -> None:
    paths = {}
    for limb in ("left_wrist", "right_wrist", "left_ankle", "right_ankle"):
        path = tmp_path / f"{limb}.npy"
        np.save(path, np.arange(30, dtype=np.float32).reshape(10, 3))
        paths[f"{limb}_path"] = str(path)
    frame = pd.DataFrame([{**paths, "participant": "P01", "label": 0, "start": -2, "stop": 8}])
    sample = WindowManifestDataset(frame, limb_mode="four", normalization="none")[0]
    assert sample["left_wrist"].shape == (10, 3)
    np.testing.assert_array_equal(
        sample["left_wrist"][:3].numpy(), np.tile(np.asarray([0, 1, 2]), (3, 1))
    )


def test_train_global_statistics_are_shared_across_sides(tmp_path: Path) -> None:
    paths = {}
    for limb, value in {
        "left_wrist": 1.0,
        "right_wrist": 3.0,
        "left_ankle": 2.0,
        "right_ankle": 4.0,
    }.items():
        path = tmp_path / f"{limb}.npy"
        np.save(path, np.full((5, 3), value, dtype=np.float32))
        paths[f"{limb}_path"] = str(path)
    frame = pd.DataFrame([{**paths, "participant": "P01", "label": 0, "start": 0, "stop": 5}])
    stats = estimate_global_normalization(frame, "four", mode="train_global")
    np.testing.assert_allclose(stats["left_wrist"]["mean"], [2.0] * 3)
    assert stats["left_wrist"] == stats["right_wrist"]
    np.testing.assert_allclose(stats["left_ankle"]["mean"], [3.0] * 3)


def test_shared_rms_ignores_invalid_samples(tmp_path: Path) -> None:
    paths = {}
    for limb in ("left_wrist", "right_wrist", "left_ankle", "right_ankle"):
        values = np.ones((4, 3), dtype=np.float32)
        values[0] = 100.0
        validity = np.ones_like(values, dtype=np.uint8)
        validity[0] = 0
        path = tmp_path / f"{limb}.npy"
        validity_path = tmp_path / f"{limb}_validity.npy"
        np.save(path, values)
        np.save(validity_path, validity)
        paths[f"{limb}_path"] = str(path)
        paths[f"{limb}_validity_path"] = str(validity_path)
    frame = pd.DataFrame([{**paths, "participant": "P01", "label": 0, "start": 0, "stop": 4}])
    stats = estimate_global_normalization(frame, "four", mode="train_rms_shared")
    np.testing.assert_allclose(stats["shared"]["rms"], [1.0])
    sample = WindowManifestDataset(
        frame,
        limb_mode="four",
        normalization="train_rms_shared",
        normalization_stats=stats,
    )[0]
    torch_missing = sample["missing_fraction"].numpy()
    np.testing.assert_allclose(torch_missing, [0.25] * 4)
