from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_realdisp_pd_pc_sensitivity import (  # noqa: E402
    SETTINGS,
    TRAIN_PARTICIPANTS,
    VALIDATION_PARTICIPANTS,
    build_setting_config,
)


def test_sensitivity_grid_is_single_factor_and_center_is_reused() -> None:
    triples = {
        (
            setting.augmentation_probability,
            setting.consistency_weight,
            setting.reuse_frozen_center,
        )
        for setting in SETTINGS
    }
    assert triples == {
        (0.25, 0.10, False),
        (0.50, 0.05, False),
        (0.50, 0.10, True),
        (0.50, 0.20, False),
        (0.75, 0.10, False),
    }
    assert TRAIN_PARTICIPANTS == tuple(f"subject{value}" for value in range(1, 11))
    assert VALIDATION_PARTICIPANTS == ("subject11", "subject12")


def test_setting_config_changes_only_declared_hyperparameters() -> None:
    base = {
        "training": {
            "quality_augmentation_probability": 0.5,
            "epochs": 50,
        },
        "loss": {
            "perturbation_prediction_consistency": 0.1,
            "role_smoothness": 0.01,
        },
    }
    setting = SETTINGS[0]
    modified = build_setting_config(base, setting)
    assert modified["training"]["quality_augmentation_probability"] == 0.25
    assert modified["loss"]["perturbation_prediction_consistency"] == 0.10
    assert modified["training"]["epochs"] == 50
    assert modified["loss"]["role_smoothness"] == 0.01
    assert base["training"]["quality_augmentation_probability"] == 0.5
