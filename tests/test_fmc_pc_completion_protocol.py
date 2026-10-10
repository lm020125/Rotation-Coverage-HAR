from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_realdisp_fmc_pc_completion import (  # noqa: E402
    AS_RELEASED_PARTICIPANTS,
    CROSS_BACKBONE,
    RCBD_ABLATIONS,
    SEEDS,
    SENSITIVITY_SETTINGS,
    TRAIN_PARTICIPANTS,
    VALIDATION_PARTICIPANTS,
    VALID_SIGNAL_PARTICIPANTS,
    evaluation_scenarios,
    sensitivity_config,
)

from rcbd_har.config import load_config  # noqa: E402


def test_completion_matrix_has_27_new_training_runs() -> None:
    non_center_sensitivity = sum(
        not setting.reuse_center for setting in SENSITIVITY_SETTINGS
    )
    total = len(SEEDS) * (
        len(RCBD_ABLATIONS) + len(CROSS_BACKBONE) + non_center_sensitivity
    )
    assert total == 27
    assert len(SEEDS) == 3


def test_test_cohorts_are_never_used_for_training_or_validation() -> None:
    development = set(TRAIN_PARTICIPANTS) | set(VALIDATION_PARTICIPANTS)
    assert not development & set(AS_RELEASED_PARTICIPANTS)
    assert not development & set(VALID_SIGNAL_PARTICIPANTS)
    assert set(VALID_SIGNAL_PARTICIPANTS) < set(AS_RELEASED_PARTICIPANTS)
    assert set(AS_RELEASED_PARTICIPANTS) - set(VALID_SIGNAL_PARTICIPANTS) == {
        "subject13"
    }


def test_evaluation_scenarios_keep_released_and_valid_signal_results() -> None:
    scenarios = evaluation_scenarios("ideal.csv", "self.csv", "mutual.csv")
    assert scenarios["as_released_self"] == (
        "self.csv",
        AS_RELEASED_PARTICIPANTS,
    )
    assert scenarios["valid_signal_self"] == (
        "self.csv",
        VALID_SIGNAL_PARTICIPANTS,
    )
    assert scenarios["subject15_mutual"] == ("mutual.csv", ("subject15",))


def test_method_configs_isolate_the_declared_fmc_components() -> None:
    expected = {
        "fmc_angle_pc": ("matched_random", True, False, False, 0.1),
        "fmc_angle_gain_pc": ("matched_random", True, True, False, 0.1),
        "fmc_full_no_pc": ("matched_random", True, True, True, 0.0),
        "fmc_global_pc": ("global_matched_random", True, True, True, 0.1),
    }
    for method, relative in RCBD_ABLATIONS.items():
        config = load_config(ROOT / relative)
        training = config["training"]
        loss = config["loss"]
        observed = (
            training["quality_augmentation_profile_sampling"],
            training["quality_augmentation_profile_rotation"],
            training["quality_augmentation_profile_gain"],
            training["quality_augmentation_profile_noise"],
            loss["perturbation_prediction_consistency"],
        )
        assert observed == expected[method]


def test_sensitivity_changes_only_probability_and_consistency() -> None:
    base = load_config(ROOT / "configs/realdisp_pd_pc_matched_random.yaml")
    setting = SENSITIVITY_SETTINGS[0]
    modified = sensitivity_config(base, setting)
    assert modified["training"]["quality_augmentation_probability"] == 0.25
    assert modified["loss"]["perturbation_prediction_consistency"] == 0.10
    assert modified["training"]["quality_augmentation_profile_sampling"] == (
        "matched_random"
    )
    assert base["training"]["quality_augmentation_probability"] == 0.50
