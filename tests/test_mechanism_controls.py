"""Numpy-only integrity tests for the frozen mechanism-control profiles.

These tests replicate the loader checks of
``rcbd_har.placement.load_placement_profile_bank`` (orthogonality, properness,
trace-derived amplitude) without requiring torch, so they can run anywhere the
profile files exist.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "profiles_mechanism_controls"
FROZEN = PROFILES / "placement_profiles_subjects1_10.json"
BETA = PROFILES / "placement_profiles_beta_moment_matched.json"
CLIP = PROFILES / "placement_profiles_clip90.json"

FROZEN_SHA = "b05eedb46aa01cff0c2331b09bc3ca4911821a0ef7c8291c0807dd19c555b2b7"
BETA_SHA = "89de4e69d5750ad1fdf37c044644248c55b31a65c265819af5619224bb1c14d0"
CLIP_SHA = "993d9ce11661af0810cc8cf77d499d034d725bdbb03b84dd86e39f919e76ad66"

LIMBS = ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
EMPIRICAL_MEAN = 40.8407099313
EMPIRICAL_SD_SAMPLE = 46.1239751073
EMPIRICAL_MAX = 175.02285456373536


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def all_records(payload: dict):
    for limb in LIMBS:
        for record in payload["limbs"][limb]:
            yield limb, record


def check_bank_geometry(payload: dict) -> list[float]:
    angles = []
    for _, record in all_records(payload):
        matrix = np.asarray(record["rotation"], dtype=np.float64)
        assert matrix.shape == (3, 3)
        assert np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6), "not orthogonal"
        assert np.linalg.det(matrix) > 0.999, "not proper"
        trace_angle = math.degrees(
            math.acos(min(1.0, max(-1.0, (np.trace(matrix) - 1.0) / 2.0)))
        )
        assert abs(trace_angle - record["angle_degrees"]) < 1e-6, (
            f"angle_degrees {record['angle_degrees']} disagrees with matrix "
            f"trace angle {trace_angle}"
        )
        angles.append(float(record["angle_degrees"]))
    return angles


def test_frozen_profile_files_are_frozen() -> None:
    assert sha256_of(FROZEN) == FROZEN_SHA
    assert sha256_of(BETA) == BETA_SHA
    assert sha256_of(CLIP) == CLIP_SHA


def test_all_variants_pass_bank_geometry() -> None:
    for path in (FROZEN, BETA, CLIP):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["protocol"]["labels_read"] is False
        angles = check_bank_geometry(payload)
        assert len(angles) == 36


def test_clip90_changes_exactly_the_five_tail_records() -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    clipped = json.loads(CLIP.read_text(encoding="utf-8"))
    changed = []
    for limb in LIMBS:
        for before, after in zip(frozen["limbs"][limb], clipped["limbs"][limb]):
            same_angle = before["angle_degrees"] == after["angle_degrees"]
            same_matrix = before["rotation"] == after["rotation"]
            assert same_angle == same_matrix, "angle and matrix must change together"
            if not same_angle:
                assert after["angle_degrees"] == 90.0
                assert before["angle_degrees"] > 90.0
                changed.append((limb, before["participant"]))
            else:
                assert before["gain"] == after["gain"]
                assert before["noise_std"] == after["noise_std"]
    assert len(changed) == 5
    assert all(angle <= 90.0 for _, angle in ((limb, r["angle_degrees"]) for limb, r in all_records(clipped)))
    participants = {participant for _, participant in changed}
    assert participants == {"subject1", "subject3", "subject4", "subject8"}


def test_beta_pool_matches_published_moments() -> None:
    payload = json.loads(BETA.read_text(encoding="utf-8"))
    angles = np.asarray([record["angle_degrees"] for _, record in all_records(payload)])
    assert abs(angles.mean() - EMPIRICAL_MEAN) < 0.5
    assert abs(angles.std(ddof=1) - EMPIRICAL_SD_SAMPLE) < 1.0
    assert angles.min() > 0.0
    assert angles.max() <= EMPIRICAL_MAX + 1e-9
    protocol = payload["protocol"]
    assert protocol["variant"] == "beta_moment_matched_amplitude"
    assert protocol["support_max_degrees"] == pytest.approx(EMPIRICAL_MAX, abs=1e-9)


def test_beta_variant_differs_from_empirical_frequencies() -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    beta = json.loads(BETA.read_text(encoding="utf-8"))
    frozen_angles = [record["angle_degrees"] for _, record in all_records(frozen)]
    beta_angles = [record["angle_degrees"] for _, record in all_records(beta)]
    assert sorted(frozen_angles) != sorted(beta_angles), (
        "control pool must not equal the empirical pool"
    )
    # Per-limb construction: within each limb (participant-ordered) the stratified
    # quantile grid is strictly increasing; it is not required to be monotone
    # across limbs because each limb is matched independently.
    for limb in LIMBS:
        records = beta["limbs"][limb]
        order = sorted(
            range(len(records)),
            key=lambda i: int(records[i]["participant"].replace("subject", "")),
        )
        seq = [records[i]["angle_degrees"] for i in order]
        assert np.all(np.diff(seq) > 0), f"{limb} quantile grid must increase"
    frozen_ranks = np.argsort(np.argsort(frozen_angles))
    assert not np.array_equal(frozen_ranks, np.arange(len(frozen_angles))), (
        "empirical pool is expected to be non-monotone in record order"
    )


def test_beta_pool_covers_tail_per_limb() -> None:
    """Regression guard for the global-sorted-slicing defect: every limb must
    retain its own large-angle tail instead of piling tail mass onto one limb."""
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    beta = json.loads(BETA.read_text(encoding="utf-8"))
    for limb in LIMBS:
        emp = np.asarray([r["angle_degrees"] for r in frozen["limbs"][limb]], dtype=float)
        bet = np.asarray([r["angle_degrees"] for r in beta["limbs"][limb]], dtype=float)
        assert bet.max() >= 0.80 * emp.max(), (
            f"{limb}: beta max {bet.max():.1f} loses empirical tail {emp.max():.1f}"
        )
        if bool((emp > 90.0).any()):
            assert bool((bet > 90.0).any()), (
                f"{limb}: empirical pool has a >90 tail but beta pool has none"
            )


def _structured_equal(left: dict, right: dict) -> None:
    """Field comparison with tight tolerances.  Rotation matrices and extreme
    quantile floats may differ by libm/scipy ulps across platforms
    (Windows/MSVC + scipy 1.17 vs Linux/glibc + scipy 1.15); the hard freeze
    remains the SHA-256 check on the archived files."""
    assert set(left) == set(right)
    for limb in LIMBS:
        assert len(left["limbs"][limb]) == len(right["limbs"][limb])
        for before, after in zip(left["limbs"][limb], right["limbs"][limb]):
            assert set(before) == set(after)
            for key, value in before.items():
                if key == "rotation":
                    assert np.allclose(
                        np.asarray(value), np.asarray(after[key]), atol=1e-12
                    ), f"{limb}/{before['participant']} rotation mismatch"
                elif isinstance(value, float):
                    assert abs(value - after[key]) < 1e-9, (
                        f"{limb}/{before['participant']}/{key}: {value} != {after[key]}"
                    )
                else:
                    assert value == after[key], f"{limb}/{before['participant']}/{key}"


def test_generator_is_deterministic() -> None:
    pytest.importorskip("scipy")
    spec = importlib.util.spec_from_file_location(
        "make_variants", ROOT / "scripts" / "make_realdisp_profile_variants.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frozen_payload = module.load_frozen(FROZEN)
    beta_variant, _ = module.beta_moment_matched_variant(frozen_payload)
    clip_variant, _ = module.clip_variant(frozen_payload)
    _structured_equal(beta_variant, json.loads(BETA.read_text(encoding="utf-8")))
    _structured_equal(clip_variant, json.loads(CLIP.read_text(encoding="utf-8")))


def test_runner_embeds_matching_freeze_hashes() -> None:
    source = (ROOT / "scripts" / "run_realdisp_mechanism_controls.py").read_text(
        encoding="utf-8"
    )
    assert FROZEN_SHA in source
    assert BETA_SHA in source
    assert CLIP_SHA in source
    assert sys.version_info >= (3, 9)
