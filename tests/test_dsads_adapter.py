from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_dsads import (  # noqa: E402
    LIMBS,
    RAW_COLUMNS,
    SEGMENT_SAMPLES,
    load_segment,
    main as prepare_dsads_main,
    parse_segment,
)
from run_grouped_cv import load_folds  # noqa: E402
from rcbd_har.config import load_config  # noqa: E402
from rcbd_har.data import WindowManifestDataset  # noqa: E402
from rcbd_har.models import build_model  # noqa: E402


def write_fake_segment(path: Path) -> np.ndarray:
    values = np.arange(
        SEGMENT_SAMPLES * RAW_COLUMNS, dtype=np.float32
    ).reshape(SEGMENT_SAMPLES, RAW_COLUMNS)
    np.savetxt(path, values, delimiter=",", fmt="%.6f")
    return values


def test_dsads_parser_and_official_accgyro_column_mapping(tmp_path: Path) -> None:
    path = tmp_path / "a19" / "p8" / "s60.txt"
    path.parent.mkdir(parents=True)
    raw = write_fake_segment(path)
    metadata = parse_segment(path)
    assert metadata["activity_id"] == 19
    assert metadata["participant"] == "p8"
    assert metadata["segment_id"] == 60
    assert metadata["label"] == 18
    arrays = load_segment(path, "acc_gyro")
    expected_starts = {
        "right_wrist": 9,
        "left_wrist": 18,
        "right_ankle": 27,
        "left_ankle": 36,
    }
    for limb, start in expected_starts.items():
        np.testing.assert_allclose(arrays[limb], raw[:, start : start + 6])


def test_dsads_anchor_maps_torso_and_writes_nine_axis_npz(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "raw"
    path = input_dir / "a01" / "p1" / "s01.txt"
    path.parent.mkdir(parents=True)
    raw = write_fake_segment(path)
    arrays = load_segment(path, "acc_gyro_mag", include_anchor=True)
    np.testing.assert_allclose(arrays["anchor"], raw[:, :9])

    output_dir = tmp_path / "processed"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_dsads.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--modalities",
            "acc_gyro_mag",
            "--include-anchor",
            "--max-files",
            "1",
        ],
    )
    prepare_dsads_main()
    manifest = pd.read_csv(output_dir / "manifest.csv")
    item = WindowManifestDataset(
        manifest, limb_mode="four", normalization="none", missing_strategy="error"
    )[0]
    assert item["anchor"].shape == (SEGMENT_SAMPLES, 9)
    assert item["missing_fraction"].shape == (5,)


def test_dsads_converter_writes_loadable_four_limb_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "raw"
    for activity, participant, segment in ((1, 1, 1), (2, 2, 1)):
        path = input_dir / f"a{activity:02d}" / f"p{participant}" / f"s{segment:02d}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_fake_segment(path)
    output_dir = tmp_path / "processed"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_dsads.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--modalities",
            "acc_gyro",
            "--max-files",
            "2",
        ],
    )
    prepare_dsads_main()
    manifest = pd.read_csv(output_dir / "manifest.csv")
    assert len(manifest) == 2
    assert set(manifest["participant"]) == {"p1", "p2"}
    dataset = WindowManifestDataset(
        manifest,
        limb_mode="four",
        normalization="none",
        missing_strategy="error",
    )
    item = dataset[0]
    for limb in LIMBS:
        assert tuple(item[limb].shape) == (SEGMENT_SAMPLES, 6)
    assert int(item["label"]) == 0


def test_dsads_config_and_loso_protocol_are_consistent() -> None:
    config = load_config(ROOT / "configs" / "dsads_rcbd.yaml")
    participants = [f"p{index}" for index in range(1, 9)]
    folds = load_folds(
        str(ROOT / "configs" / "dsads_loso_folds.json"), participants, n_splits=8
    )
    assert folds == [[participant] for participant in participants]
    assert config["data"]["imu_channels"] == 6
    assert config["data"]["num_classes"] == 19
    assert config["evaluation"]["enabled"] is False
    assert config["cross_validation"]["val_participants_per_fold"] == 1


def test_dsads_rcbd_config_is_swap_invariant_with_six_axis_input() -> None:
    config = load_config(ROOT / "configs" / "dsads_rcbd.yaml")
    model = build_model(config).eval()
    limbs = [
        np.random.default_rng(index).normal(size=(2, SEGMENT_SAMPLES, 6)).astype(np.float32)
        for index in range(4)
    ]
    tensors = [torch.from_numpy(values) for values in limbs]
    with torch.no_grad():
        reference = model(*tensors)["logits"]
        arm_swapped = model(tensors[1], tensors[0], tensors[2], tensors[3])["logits"]
        leg_swapped = model(tensors[0], tensors[1], tensors[3], tensors[2])["logits"]
    torch.testing.assert_close(reference, arm_swapped, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(reference, leg_swapped, atol=1e-6, rtol=1e-5)


def test_dsads_arrcbd_config_builds_and_is_swap_invariant_with_anchor() -> None:
    config = load_config(ROOT / "configs" / "dsads_arrcbd.yaml")
    assert config["data"]["imu_channels"] == 9
    assert config["model"]["anchor_mode"] == "referenced"
    model = build_model(config).eval()
    tensors = [torch.randn(2, SEGMENT_SAMPLES, 9) for _ in range(4)]
    anchor = torch.randn(2, SEGMENT_SAMPLES, 9)
    with torch.no_grad():
        reference = model(*tensors, anchor=anchor)["logits"]
        arm_swapped = model(
            tensors[1], tensors[0], tensors[2], tensors[3], anchor=anchor
        )["logits"]
        leg_swapped = model(
            tensors[0], tensors[1], tensors[3], tensors[2], anchor=anchor
        )["logits"]
    torch.testing.assert_close(reference, arm_swapped, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(reference, leg_swapped, atol=1e-6, rtol=1e-5)


def test_dsads_selective_dual_config_builds_both_experts() -> None:
    config = load_config(ROOT / "configs" / "dsads_selective_dual_rcbd.yaml")
    assert config["data"]["imu_channels"] == 9
    assert config["model"]["anchor_mode"] == "selective_dual"
    model = build_model(config).eval()
    tensors = [torch.randn(2, SEGMENT_SAMPLES, 9) for _ in range(4)]
    anchor = torch.randn(2, SEGMENT_SAMPLES, 9)
    with torch.no_grad():
        output = model(*tensors, anchor=anchor)
    assert output["logits"].shape == (2, 19)
    assert output["ordered_logits"].shape == (2, 19)
    assert output["safe_logits"].shape == (2, 19)


def test_dsads_decoupled_selective_config_uses_independent_encoders() -> None:
    config = load_config(ROOT / "configs" / "dsads_selective_decoupled_rcbd.yaml")
    assert config["model"]["dual_encoder_sharing"] is False
    model = build_model(config).eval()
    assert model.arm_encoder is not model.arm_ordered_encoder
    assert model.leg_encoder is not model.leg_ordered_encoder
    assert model.anchor_encoder is not model.anchor_ordered_encoder
    safe_ids = {id(parameter) for parameter in model.arm_encoder.parameters()}
    ordered_ids = {id(parameter) for parameter in model.arm_ordered_encoder.parameters()}
    assert safe_ids.isdisjoint(ordered_ids)

    tensors = [torch.randn(2, SEGMENT_SAMPLES, 9) for _ in range(4)]
    anchor = torch.randn(2, SEGMENT_SAMPLES, 9)
    with torch.no_grad():
        output = model(*tensors, anchor=anchor)
        swapped = model(tensors[1], tensors[0], tensors[3], tensors[2], anchor=anchor)
    torch.testing.assert_close(output["safe_logits"], swapped["safe_logits"], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(output["ordered_logits"], swapped["ordered_logits"])


def test_dsads_hard_route_selects_one_expert_without_logit_contamination() -> None:
    config = load_config(ROOT / "configs" / "dsads_selective_hard_decoupled_rcbd.yaml")
    model = build_model(config).eval()
    tensors = [torch.randn(2, SEGMENT_SAMPLES, 9) for _ in range(4)]
    anchor = torch.randn(2, SEGMENT_SAMPLES, 9)
    with torch.no_grad():
        ordered = model(*tensors, anchor=anchor)
    torch.testing.assert_close(ordered["ordered_route"], torch.ones_like(ordered["ordered_route"]))
    torch.testing.assert_close(ordered["logits"], ordered["ordered_logits"], atol=0, rtol=0)

    with torch.no_grad():
        model.arm_side_trust[-1].bias.fill_(-2.0)
        model.leg_side_trust[-1].bias.fill_(-2.0)
        safe = model(*tensors, anchor=anchor)
    torch.testing.assert_close(safe["ordered_route"], torch.zeros_like(safe["ordered_route"]))
    torch.testing.assert_close(safe["logits"], safe["safe_logits"], atol=0, rtol=0)
