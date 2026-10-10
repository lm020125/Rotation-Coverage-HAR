from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rcbd_har.data import WindowManifestDataset, estimate_global_normalization, load_manifest
from rcbd_har.engine import forward_batch
from rcbd_har.models import build_model
from rcbd_har.models.deepconvlstm import DeepConvLSTM, FourLimbDeepConvLSTM


def test_continuous_bilateral_manifest_slices_memmapped_arrays(tmp_path: Path) -> None:
    left = np.arange(120, dtype=np.float32).reshape(20, 6)
    right = left + 1_000
    valid = np.ones_like(left, dtype=np.uint8)
    paths = {}
    for name, values in {"left": left, "right": right}.items():
        paths[name] = tmp_path / f"{name}.npy"
        np.save(paths[name], values)
        np.save(tmp_path / f"{name}_validity.npy", valid)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "participant": "subject1",
                "label": 2,
                "start": 2,
                "stop": 12,
                "center": 7,
                "left_path": paths["left"],
                "right_path": paths["right"],
                "left_validity_path": tmp_path / "left_validity.npy",
                "right_validity_path": tmp_path / "right_validity.npy",
            }
        ]
    ).to_csv(manifest_path, index=False)
    frame = load_manifest(manifest_path)
    stats = estimate_global_normalization(frame, "bilateral", mode="train_global")
    dataset = WindowManifestDataset(
        frame,
        normalization="train_global",
        normalization_stats=stats,
        limb_mode="bilateral",
    )
    item = dataset[0]
    assert item["left"].shape == (10, 6)
    assert item["right"].shape == (10, 6)
    assert int(item["label"]) == 2
    assert torch.isfinite(item["left"]).all()


def test_deepconvlstm_published_topology_output_shape() -> None:
    model = DeepConvLSTM(input_channels_per_sensor=6, num_classes=33)
    left = torch.randn(4, 100, 6)
    right = torch.randn(4, 100, 6)
    output = model(left, right)
    assert output["embedding"].shape == (4, 128)
    assert output["logits"].shape == (4, 33)


def test_four_limb_deepconvlstm_preserves_published_backbone() -> None:
    model = FourLimbDeepConvLSTM(input_channels_per_sensor=3, num_classes=33)
    sensors = [torch.randn(4, 400, 3) for _ in range(4)]
    output = model(*sensors)
    assert output["embedding"].shape == (4, 128)
    assert output["logits"].shape == (4, 33)
    assert model.convolutions[0].in_channels == 12


def test_build_four_limb_deepconvlstm_uses_generic_batch_dispatch() -> None:
    config = {
        "data": {"limb_mode": "four", "imu_channels": 3, "num_classes": 33},
        "model": {"architecture": "deepconvlstm", "variant": "concat"},
    }
    model = build_model(config)
    batch = {
        key: torch.randn(2, 400, 3)
        for key in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
    }
    output = forward_batch(model, batch)
    assert isinstance(model, FourLimbDeepConvLSTM)
    assert output["logits"].shape == (2, 33)
