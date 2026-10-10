from __future__ import annotations

import json

import numpy as np
import torch

from rcbd_har.placement import (
    _gravity_tilt_rotations,
    _random_axis_rotations,
    _sample_profile_components,
    apply_calibrated_one_side_per_pair,
    estimate_placement_profile,
    load_placement_profile_bank,
    shortest_arc_rotation,
)


def test_gravity_tilt_rotations_match_requested_actual_tilt() -> None:
    torch.manual_seed(123)
    gravity = torch.randn(512, 3)
    angles = torch.linspace(0.0, float(np.pi * 0.95), len(gravity))

    rotations = _gravity_tilt_rotations(angles, gravity, torch.float32)
    rotated = torch.einsum("bij,bj->bi", rotations, gravity)
    cosine = (
        (gravity * rotated).sum(dim=-1)
        / (gravity.norm(dim=-1) * rotated.norm(dim=-1)).clamp_min(1e-8)
    ).clamp(-1.0, 1.0)
    actual = torch.acos(cosine)

    assert torch.allclose(actual, angles, atol=5e-4)
    identity = torch.eye(3).expand_as(rotations)
    assert torch.allclose(rotations.transpose(-1, -2) @ rotations, identity, atol=1e-5)
    assert torch.allclose(torch.linalg.det(rotations), torch.ones(len(gravity)), atol=1e-5)


def test_shortest_arc_rotation_maps_gravity_and_is_proper() -> None:
    source = np.asarray([0.0, 0.0, 1.0])
    target = np.asarray([1.0, 0.0, 0.0])
    rotation = shortest_arc_rotation(source, target)

    np.testing.assert_allclose(rotation @ source, target, atol=1e-7)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-7)


def test_shortest_arc_rotation_handles_opposite_gravity() -> None:
    source = np.asarray([0.0, 0.0, 1.0])
    target = -source
    rotation = shortest_arc_rotation(source, target)

    np.testing.assert_allclose(rotation @ source, target, atol=1e-7)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-7)


def test_profile_estimation_rejects_zero_filled_self_sensor() -> None:
    ideal = np.ones((20, 3), dtype=np.float32)
    self_placed = np.zeros((20, 3), dtype=np.float32)

    try:
        estimate_placement_profile(ideal, self_placed)
    except ValueError as error:
        assert "zero-filled" in str(error)
    else:
        raise AssertionError("zero-filled placement data should be rejected")


def test_profile_estimation_recovers_tilt_and_gain() -> None:
    rng = np.random.default_rng(2)
    ideal = rng.normal(scale=0.1, size=(1000, 3)) + np.asarray([0.0, 0.0, 9.8])
    expected_rotation = shortest_arc_rotation(
        np.asarray([0.0, 0.0, 1.0]), np.asarray([1.0, 0.0, 0.0])
    )
    self_placed = 1.1 * np.einsum("ij,tj->ti", expected_rotation, ideal)

    profile = estimate_placement_profile(ideal, self_placed)

    np.testing.assert_allclose(profile["rotation"], expected_rotation, atol=2e-3)
    np.testing.assert_allclose(profile["gain"], 1.1, atol=1e-3)
    np.testing.assert_allclose(profile["angle_degrees"], 90.0, atol=0.2)


def test_calibrated_augmentation_uses_profiles_and_preserves_one_side(tmp_path) -> None:
    rotation = shortest_arc_rotation(
        np.asarray([0.0, 0.0, 1.0]), np.asarray([1.0, 0.0, 0.0])
    ).tolist()
    payload = {
        "protocol": {"labels_read": False},
        "limbs": {
            limb: [{"rotation": rotation, "gain": 2.0, "noise_std": 0.0}]
            for limb in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
        },
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    bank = load_placement_profile_bank(
        path, device=torch.device("cpu"), dtype=torch.float32, shared_rms=1.0
    )
    batch = {
        limb: torch.tensor([[[0.0, 0.0, 1.0]]])
        for limb in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
    }

    torch.manual_seed(4)
    augmented, _ = apply_calibrated_one_side_per_pair(batch, bank, probability=1.0)

    expected = torch.tensor([[[2.0, 0.0, 0.0]]])
    for prefix, pair in {
        "arm": ("left_wrist", "right_wrist"),
        "leg": ("left_ankle", "right_ankle"),
    }.items():
        changed = [not torch.equal(augmented[key], batch[key]) for key in pair]
        assert sum(changed) == 1
        changed_key = pair[changed.index(True)]
        assert torch.allclose(augmented[changed_key], expected, atol=1e-6)
        assert bool(
            (augmented[f"{prefix}_rotated_left"] | augmented[f"{prefix}_rotated_right"]).item()
        )


def test_calibrated_augmentation_supports_bilateral_arm_batches(tmp_path) -> None:
    rotation = shortest_arc_rotation(
        np.asarray([0.0, 0.0, 1.0]), np.asarray([1.0, 0.0, 0.0])
    ).tolist()
    payload = {
        "protocol": {"labels_read": False},
        "limbs": {
            limb: [{"rotation": rotation, "gain": 1.0, "noise_std": 0.0}]
            for limb in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
        },
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    bank = load_placement_profile_bank(
        path, device=torch.device("cpu"), dtype=torch.float32, shared_rms=1.0
    )
    batch = {
        "left": torch.tensor([[[0.0, 0.0, 1.0]]]),
        "right": torch.tensor([[[0.0, 0.0, 1.0]]]),
    }

    torch.manual_seed(4)
    augmented, targets = apply_calibrated_one_side_per_pair(
        batch,
        bank,
        probability=1.0,
        use_rotation=True,
        use_gain=False,
        use_noise=False,
    )

    changed = [not torch.equal(augmented[key], batch[key]) for key in ("left", "right")]
    assert sum(changed) == 1
    assert set(targets) == {"arm", "arm_severity_radians"}
    assert bool((augmented["arm_rotated_left"] | augmented["arm_rotated_right"]).item())


def test_bilateral_profile_loader_accepts_only_arm_profiles(tmp_path) -> None:
    payload = {
        "protocol": {"labels_read": False},
        "limbs": {
            limb: [
                {
                    "rotation": np.eye(3).tolist(),
                    "gain": 1.0,
                    "noise_std": 0.0,
                    "participant": "S04",
                }
            ]
            for limb in ("left_wrist", "right_wrist")
        },
    }
    path = tmp_path / "arms_only.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    bank = load_placement_profile_bank(
        path,
        device=torch.device("cpu"),
        dtype=torch.float32,
        shared_rms=1.0,
        required_limbs=("left_wrist", "right_wrist"),
    )

    assert set(bank) == {"left_wrist", "right_wrist"}


def test_participant_balanced_profile_sampling_ignores_record_count(tmp_path) -> None:
    entries = [
        {
            "rotation": np.eye(3).tolist(),
            "gain": gain,
            "noise_std": 0.0,
            "participant": participant,
        }
        for participant, gain in (("S04", 2.0), ("S05", 5.0), ("S05", 6.0), ("S05", 7.0))
    ]
    payload = {
        "protocol": {"labels_read": False},
        "limbs": {limb: entries for limb in ("left_wrist", "right_wrist")},
    }
    path = tmp_path / "unbalanced.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    bank = load_placement_profile_bank(
        path,
        device=torch.device("cpu"),
        dtype=torch.float32,
        shared_rms=1.0,
        required_limbs=("left_wrist", "right_wrist"),
    )

    torch.manual_seed(91)
    _, gain, _, _ = _sample_profile_components(
        bank,
        "left_wrist",
        4000,
        "joint",
        participant_balanced=True,
    )

    s04_fraction = float((gain == 2.0).float().mean().item())
    assert 0.46 <= s04_fraction <= 0.54


def test_calibrated_component_flags_isolate_rotation_gain_and_noise(tmp_path) -> None:
    rotation = shortest_arc_rotation(
        np.asarray([0.0, 0.0, 1.0]), np.asarray([1.0, 0.0, 0.0])
    ).tolist()
    payload = {
        "protocol": {"labels_read": False},
        "limbs": {
            limb: [{"rotation": rotation, "gain": 2.0, "noise_std": 0.0}]
            for limb in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
        },
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    bank = load_placement_profile_bank(
        path, device=torch.device("cpu"), dtype=torch.float32, shared_rms=1.0
    )
    batch = {
        limb: torch.tensor([[[0.0, 0.0, 1.0]]])
        for limb in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
    }

    torch.manual_seed(4)
    rotation_only, _ = apply_calibrated_one_side_per_pair(
        batch,
        bank,
        probability=1.0,
        use_rotation=True,
        use_gain=False,
        use_noise=False,
    )
    torch.manual_seed(4)
    rotation_gain, _ = apply_calibrated_one_side_per_pair(
        batch,
        bank,
        probability=1.0,
        use_rotation=True,
        use_gain=True,
        use_noise=False,
    )

    for pair in (("left_wrist", "right_wrist"), ("left_ankle", "right_ankle")):
        changed = [not torch.equal(rotation_only[key], batch[key]) for key in pair]
        changed_key = pair[changed.index(True)]
        assert torch.allclose(
            rotation_only[changed_key], torch.tensor([[[1.0, 0.0, 0.0]]]), atol=1e-6
        )
        assert torch.allclose(
            rotation_gain[changed_key], torch.tensor([[[2.0, 0.0, 0.0]]]), atol=1e-6
        )


def test_calibrated_component_flags_reject_empty_transform(tmp_path) -> None:
    payload = {
        "protocol": {"labels_read": False},
        "limbs": {
            limb: [{"rotation": np.eye(3).tolist(), "gain": 1.0, "noise_std": 0.0}]
            for limb in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
        },
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    bank = load_placement_profile_bank(
        path, device=torch.device("cpu"), dtype=torch.float32, shared_rms=1.0
    )
    batch = {
        limb: torch.ones((1, 1, 3))
        for limb in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
    }

    try:
        apply_calibrated_one_side_per_pair(
            batch,
            bank,
            probability=1.0,
            use_rotation=False,
            use_gain=False,
            use_noise=False,
        )
    except ValueError as error:
        assert "At least one" in str(error)
    else:
        raise AssertionError("An empty calibrated transform should be rejected")


def test_matched_random_uses_proper_rotations_and_preserves_empirical_gain(tmp_path) -> None:
    quarter_turn = shortest_arc_rotation(
        np.asarray([0.0, 0.0, 1.0]), np.asarray([1.0, 0.0, 0.0])
    ).tolist()
    payload = {
        "protocol": {"labels_read": False},
        "limbs": {
            limb: [
                {"rotation": np.eye(3).tolist(), "gain": 1.0, "noise_std": 0.0},
                {"rotation": quarter_turn, "gain": 2.0, "noise_std": 0.0},
            ]
            for limb in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
        },
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    bank = load_placement_profile_bank(
        path, device=torch.device("cpu"), dtype=torch.float32, shared_rms=1.0
    )
    batch = {
        limb: torch.tensor([[[0.0, 0.0, 1.0]]] * 128)
        for limb in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
    }

    torch.manual_seed(17)
    augmented, _ = apply_calibrated_one_side_per_pair(
        batch, bank, probability=1.0, sampling_mode="matched_random"
    )

    for prefix, pair in {
        "arm": ("left_wrist", "right_wrist"),
        "leg": ("left_ankle", "right_ankle"),
    }.items():
        selected = torch.where(
            augmented[f"{prefix}_rotated_left"].view(-1, 1, 1),
            augmented[pair[0]],
            augmented[pair[1]],
        )
        norms = selected.norm(dim=-1).flatten()
        assert set(torch.round(norms, decimals=5).tolist()).issubset({1.0, 2.0})
        assert len(torch.unique(torch.round(norms, decimals=5))) == 2

    sampled_angles = torch.tensor([0.0, np.pi / 6.0, np.pi / 2.0], dtype=torch.float32)
    rotations = _random_axis_rotations(sampled_angles, torch.float32)
    identity = torch.eye(3).expand_as(rotations)
    assert torch.allclose(rotations.transpose(-1, -2) @ rotations, identity, atol=1e-5)
    assert torch.allclose(torch.linalg.det(rotations), torch.ones(3), atol=1e-5)
    recovered = torch.acos(
        ((rotations.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    )
    assert torch.allclose(recovered, sampled_angles, atol=1e-5)


def test_limb_shuffle_never_uses_target_limbs_only_profile(tmp_path) -> None:
    gains = {
        "left_wrist": 2.0,
        "right_wrist": 3.0,
        "left_ankle": 4.0,
        "right_ankle": 5.0,
    }
    payload = {
        "protocol": {"labels_read": False},
        "limbs": {
            limb: [{"rotation": np.eye(3).tolist(), "gain": gain, "noise_std": 0.0}]
            for limb, gain in gains.items()
        },
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    bank = load_placement_profile_bank(
        path, device=torch.device("cpu"), dtype=torch.float32, shared_rms=1.0
    )
    batch = {
        limb: torch.ones((64, 1, 3))
        for limb in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
    }

    torch.manual_seed(23)
    augmented, _ = apply_calibrated_one_side_per_pair(
        batch,
        bank,
        probability=1.0,
        use_rotation=False,
        use_gain=True,
        use_noise=False,
        sampling_mode="limb_shuffle",
    )

    for prefix, pair in {
        "arm": ("left_wrist", "right_wrist"),
        "leg": ("left_ankle", "right_ankle"),
    }.items():
        for key, mask_key in zip(
            pair, (f"{prefix}_rotated_left", f"{prefix}_rotated_right"), strict=True
        ):
            selected = augmented[mask_key]
            if bool(selected.any().item()):
                observed_gain = augmented[key][selected, 0, 0]
                assert bool((observed_gain != gains[key]).all().item())


def test_global_matched_random_pools_all_limb_marginals(tmp_path) -> None:
    gains = {
        "left_wrist": 2.0,
        "right_wrist": 3.0,
        "left_ankle": 4.0,
        "right_ankle": 5.0,
    }
    payload = {
        "protocol": {"labels_read": False},
        "limbs": {
            limb: [{"rotation": np.eye(3).tolist(), "gain": gain, "noise_std": gain / 10.0}]
            for limb, gain in gains.items()
        },
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    bank = load_placement_profile_bank(
        path, device=torch.device("cpu"), dtype=torch.float32, shared_rms=1.0
    )

    torch.manual_seed(71)
    _, sampled_gain, sampled_noise, _ = _sample_profile_components(
        bank, "left_wrist", 2048, "global_matched_random"
    )

    assert set(torch.unique(sampled_gain).tolist()) == set(gains.values())
    np.testing.assert_allclose(
        np.sort(torch.unique(sampled_noise).numpy()),
        np.asarray([0.2, 0.3, 0.4, 0.5]),
        atol=1e-6,
    )
    assert bool((sampled_gain != gains["left_wrist"]).any().item())


def test_profile_negative_controls_preserve_declared_marginals_and_tuple_coupling(
    tmp_path,
) -> None:
    quarter_turn = shortest_arc_rotation(
        np.asarray([0.0, 0.0, 1.0]), np.asarray([1.0, 0.0, 0.0])
    ).tolist()
    limb_values = {
        "left_wrist": (2.0, 0.2),
        "right_wrist": (3.0, 0.3),
        "left_ankle": (4.0, 0.4),
        "right_ankle": (5.0, 0.5),
    }
    payload = {
        "protocol": {"labels_read": False},
        "limbs": {
            limb: [
                {"rotation": np.eye(3).tolist(), "gain": 1.0, "noise_std": 0.1},
                {"rotation": quarter_turn, "gain": gain, "noise_std": noise},
            ]
            for limb, (gain, noise) in limb_values.items()
        },
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    bank = load_placement_profile_bank(
        path, device=torch.device("cpu"), dtype=torch.float32, shared_rms=1.0
    )

    torch.manual_seed(41)
    rotations, sampled_gain, sampled_noise, _ = _sample_profile_components(
        bank, "left_wrist", 1024, "matched_random"
    )
    recovered_angles = torch.acos(
        ((rotations.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    )
    assert bool(
        (
            torch.isclose(recovered_angles, torch.zeros_like(recovered_angles), atol=1e-5)
            | torch.isclose(
                recovered_angles,
                torch.full_like(recovered_angles, float(np.pi / 2.0)),
                atol=1e-5,
            )
        )
        .all()
        .item()
    )
    assert set(torch.round(sampled_gain, decimals=5).tolist()) == {1.0, 2.0}
    np.testing.assert_allclose(
        np.sort(torch.unique(sampled_noise).cpu().numpy()),
        np.asarray([0.1, 0.2]),
        atol=1e-6,
    )
    assert bool(((recovered_angles < 1e-4) & (sampled_gain > 1.5)).any().item())
    assert bool(((recovered_angles > 1.0) & (sampled_gain < 1.5)).any().item())

    torch.manual_seed(43)
    _, wrong_gain, wrong_noise, _ = _sample_profile_components(
        bank, "left_wrist", 1024, "limb_shuffle"
    )
    non_identity = wrong_gain > 1.5
    assert bool(non_identity.any().item())
    assert bool(torch.allclose(wrong_gain[non_identity], 10.0 * wrong_noise[non_identity]))
    assert not bool((wrong_gain == limb_values["left_wrist"][0]).any().item())


def test_calibrated_augmentation_rejects_unknown_sampling_mode(tmp_path) -> None:
    payload = {
        "protocol": {"labels_read": False},
        "limbs": {
            limb: [{"rotation": np.eye(3).tolist(), "gain": 1.0, "noise_std": 0.0}]
            for limb in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
        },
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    bank = load_placement_profile_bank(
        path, device=torch.device("cpu"), dtype=torch.float32, shared_rms=1.0
    )
    batch = {
        limb: torch.ones((1, 1, 3))
        for limb in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
    }

    try:
        apply_calibrated_one_side_per_pair(
            batch, bank, probability=1.0, sampling_mode="participant_shuffle"
        )
    except ValueError as error:
        assert "Unknown placement profile sampling mode" in str(error)
    else:
        raise AssertionError("Unknown profile sampling modes must be rejected")


def test_participant_exclusive_sampling_uses_only_other_profiles(tmp_path) -> None:
    payload = {
        "protocol": {"labels_read": False},
        "limbs": {
            limb: [
                {
                    "rotation": np.eye(3).tolist(),
                    "gain": 2.0,
                    "noise_std": 0.0,
                    "participant": "subject1",
                },
                {
                    "rotation": np.eye(3).tolist(),
                    "gain": 3.0,
                    "noise_std": 0.0,
                    "participant": "subject2",
                },
            ]
            for limb in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
        },
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    bank = load_placement_profile_bank(
        path, device=torch.device("cpu"), dtype=torch.float32, shared_rms=1.0
    )
    batch = {
        **{
            limb: torch.ones((16, 1, 3))
            for limb in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
        },
        "participant": ["subject1"] * 16,
    }

    augmented, _ = apply_calibrated_one_side_per_pair(
        batch,
        bank,
        probability=1.0,
        use_rotation=False,
        use_gain=True,
        use_noise=False,
        sampling_mode="joint",
        exclude_source_participant=True,
    )

    for prefix, pair in {
        "arm": ("left_wrist", "right_wrist"),
        "leg": ("left_ankle", "right_ankle"),
    }.items():
        selected = torch.where(
            augmented[f"{prefix}_rotated_left"].view(-1, 1, 1),
            augmented[pair[0]],
            augmented[pair[1]],
        )
        assert torch.allclose(selected, torch.full_like(selected, 3.0))
