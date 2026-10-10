from __future__ import annotations

import json
from pathlib import Path

import pytest

from rcbd_har.engine import placement_profile_raw_scale
from scripts.run_wear_fmc_pc_cv import METHODS, condition_config, validate_profile


def _base() -> dict:
    return {
        "seed": 1,
        "data": {"normalization": "scale_only", "normalization_scale": 8.0},
        "training": {},
        "loss": {},
    }


def _profile(path: Path) -> None:
    payload = {
        "protocol": {"labels_read": False},
        "limbs": {
            limb: [
                {
                    "participant": "subject_1",
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "gain": 1.0,
                    "noise_std": 0.0,
                }
            ]
            for limb in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_wear_conditions_are_cleanly_separated(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    _profile(profile)
    for method in METHODS:
        config = condition_config(_base(), method, 3407, profile)
        training = config["training"]
        pc = config["loss"]["perturbation_prediction_consistency"]
        assert config["seed"] == 3407
        if method == "clean":
            assert training["quality_augmentation_probability"] == 0.0
        if method == "lr_swap":
            assert training["swap_augmentation_probability"] == 0.5
        if method.startswith("uniform180"):
            assert training["quality_augmentation_max_degrees"] == 180.0
        if method.startswith("fmc"):
            assert training["quality_augmentation_profile_path"] == str(profile.resolve())
            assert training["quality_augmentation_profile_sampling"] == "gravity_matched_random"
        assert (pc == 0.1) == (method in {"uniform180_pc", "fmc_pc"})


def test_profile_rejects_external_wear_participants(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    _profile(profile)
    validate_profile(profile)
    payload = json.loads(profile.read_text(encoding="utf-8"))
    payload["limbs"]["left_wrist"][0]["participant"] = "sbj_18"
    profile.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="External WEAR"):
        validate_profile(profile)


def test_calibrated_profile_supports_wear_scale_only() -> None:
    assert placement_profile_raw_scale(_base()) == 8.0
    with pytest.raises(ValueError, match="positive raw scale"):
        placement_profile_raw_scale(
            {"data": {"normalization": "scale_only", "normalization_scale": 0.0}}
        )
