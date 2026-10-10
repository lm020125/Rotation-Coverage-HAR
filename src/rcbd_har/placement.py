from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


LIMB_PAIRS = {
    "arm": ("left_wrist", "right_wrist"),
    "leg": ("left_ankle", "right_ankle"),
}
LIMB_KEYS = tuple(key for pair in LIMB_PAIRS.values() for key in pair)
PROFILE_SAMPLING_MODES = (
    "joint",
    "matched_random",
    "global_matched_random",
    "gravity_matched_random",
    "global_gravity_matched_random",
    "limb_shuffle",
)


def shortest_arc_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the minimum-angle proper rotation mapping one direction to another."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != (3,) or target.shape != (3,):
        raise ValueError("Gravity directions must have shape [3]")
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if min(source_norm, target_norm) <= 1e-8:
        raise ValueError("Cannot estimate placement rotation from a zero gravity direction")
    a = source / source_norm
    b = target / target_norm
    cross = np.cross(a, b)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if sine <= 1e-8:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        basis = np.zeros(3, dtype=np.float64)
        basis[int(np.argmin(np.abs(a)))] = 1.0
        axis = np.cross(a, basis)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
    axis = cross / sine
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        cosine * np.eye(3, dtype=np.float64) + (1.0 - cosine) * np.outer(axis, axis) + sine * skew
    )


def _valid_rows(values: np.ndarray, validity: np.ndarray | None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Placement calibration requires [N,3] arrays, got {values.shape}")
    valid = np.isfinite(values).all(axis=1)
    if validity is not None:
        validity = np.asarray(validity, dtype=bool)
        if validity.shape != values.shape:
            raise ValueError(
                f"Validity shape {validity.shape} does not match values {values.shape}"
            )
        valid &= validity.all(axis=1)
    return values[valid]


def estimate_placement_profile(
    ideal: np.ndarray,
    self_placed: np.ndarray,
    *,
    ideal_validity: np.ndarray | None = None,
    self_validity: np.ndarray | None = None,
    maximum_zero_fraction: float = 0.5,
    gain_limits: tuple[float, float] = (0.75, 1.25),
    maximum_noise_fraction: float = 0.05,
) -> dict[str, Any]:
    """Estimate a label-free, acceleration-identifiable Ideal-to-Self profile."""
    ideal_values = _valid_rows(ideal, ideal_validity)
    self_values = _valid_rows(self_placed, self_validity)
    if min(len(ideal_values), len(self_values)) < 2:
        raise ValueError("Placement calibration requires at least two valid samples per domain")
    ideal_norm = np.linalg.norm(ideal_values, axis=1)
    self_norm = np.linalg.norm(self_values, axis=1)
    ideal_zero_fraction = float(np.mean(ideal_norm <= 1e-6))
    self_zero_fraction = float(np.mean(self_norm <= 1e-6))
    if max(ideal_zero_fraction, self_zero_fraction) > maximum_zero_fraction:
        raise ValueError(
            "Sensor is effectively zero-filled: "
            f"ideal={ideal_zero_fraction:.3f}, self={self_zero_fraction:.3f}"
        )
    ideal_mean = ideal_values.mean(axis=0)
    self_mean = self_values.mean(axis=0)
    rotation = shortest_arc_rotation(ideal_mean, self_mean)
    cosine = float(
        np.clip(
            np.dot(ideal_mean, self_mean)
            / max(np.linalg.norm(ideal_mean) * np.linalg.norm(self_mean), 1e-12),
            -1.0,
            1.0,
        )
    )
    angle_degrees = float(np.degrees(np.arccos(cosine)))
    ideal_rms = float(np.sqrt(np.mean(ideal_norm**2)))
    self_rms = float(np.sqrt(np.mean(self_norm**2)))
    raw_gain = self_rms / max(ideal_rms, 1e-8)
    gain = float(np.clip(raw_gain, *gain_limits))
    ideal_difference = np.diff(ideal_values, axis=0)
    self_difference = np.diff(self_values, axis=0)
    ideal_difference_scale = float(np.median(np.linalg.norm(ideal_difference, axis=1)))
    self_difference_scale = float(np.median(np.linalg.norm(self_difference, axis=1)))
    excess_difference = max(self_difference_scale - gain * ideal_difference_scale, 0.0)
    noise_std = min(excess_difference / np.sqrt(2.0), maximum_noise_fraction * ideal_rms)
    return {
        "rotation": rotation.tolist(),
        "angle_degrees": angle_degrees,
        "gain": gain,
        "raw_gain": raw_gain,
        "noise_std": float(noise_std),
        "ideal_rms": ideal_rms,
        "self_rms": self_rms,
        "ideal_zero_fraction": ideal_zero_fraction,
        "self_zero_fraction": self_zero_fraction,
        "ideal_valid_samples": int(len(ideal_values)),
        "self_valid_samples": int(len(self_values)),
    }


def load_placement_profile_bank(
    path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    shared_rms: float,
    required_limbs: Sequence[str] = LIMB_KEYS,
) -> dict[str, dict[str, Any]]:
    """Load audited profile JSON and convert raw-unit noise to normalized units."""
    if shared_rms <= 0:
        raise ValueError("shared_rms must be positive")
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    protocol = payload.get("protocol", {})
    if protocol.get("labels_read") is not False:
        raise ValueError("Placement profile must explicitly attest labels_read=false")
    limbs = payload.get("limbs", {})
    bank: dict[str, dict[str, Any]] = {}
    required_limbs = tuple(str(limb) for limb in required_limbs)
    if not required_limbs or not set(required_limbs).issubset(LIMB_KEYS):
        raise ValueError(f"Invalid required placement limbs: {required_limbs}")
    for limb in required_limbs:
        profiles = limbs.get(limb, [])
        if not profiles:
            raise ValueError(f"Placement profile bank is empty for {limb}")
        rotations = torch.as_tensor(
            [profile["rotation"] for profile in profiles], device=device, dtype=dtype
        )
        gains = torch.as_tensor(
            [profile["gain"] for profile in profiles], device=device, dtype=dtype
        )
        noise = torch.as_tensor(
            [profile["noise_std"] / shared_rms for profile in profiles],
            device=device,
            dtype=dtype,
        )
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
            raise ValueError(f"Invalid rotation bank for {limb}: {rotations.shape}")
        identity = torch.eye(3, device=device, dtype=dtype)
        orthogonality = rotations.transpose(-1, -2) @ rotations
        if not torch.allclose(orthogonality, identity.expand_as(orthogonality), atol=1e-4):
            raise ValueError(f"Placement rotations are not orthogonal for {limb}")
        if bool((torch.linalg.det(rotations) < 0.999).any().item()):
            raise ValueError(f"Placement rotations must be proper for {limb}")
        traces = rotations.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        angles = torch.acos(((traces - 1.0) / 2.0).clamp(-1.0, 1.0))
        bank[limb] = {
            "rotation": rotations,
            "angle_radians": angles,
            "gain": gains,
            "noise_std": noise,
            "participant": tuple(str(profile.get("participant", "")) for profile in profiles),
        }
    return bank


def _random_axis_rotations(angles: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Generate proper SO(3) rotations with fixed angles and isotropic random axes."""
    axes = torch.randn(angles.shape[0], 3, device=angles.device, dtype=dtype)
    axes = axes / axes.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z = axes.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack((zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=-1).reshape(-1, 3, 3)
    identity = torch.eye(3, device=angles.device, dtype=dtype).expand(len(angles), -1, -1)
    outer = axes.unsqueeze(-1) * axes.unsqueeze(-2)
    cosine = angles.to(dtype=dtype).cos().view(-1, 1, 1)
    sine = angles.to(dtype=dtype).sin().view(-1, 1, 1)
    return cosine * identity + (1.0 - cosine) * outer + sine * skew


def _gravity_tilt_rotations(
    angles: torch.Tensor,
    gravity: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Rotate each gravity direction by exactly the requested tilt angle.

    A random rotation axis is sampled uniformly in the tangent circle orthogonal
    to the estimated gravity direction.  Consequently, the geodesic angle
    between ``gravity`` and ``R @ gravity`` equals ``angles`` (up to numerical
    precision), unlike an unconstrained random-axis rotation.
    """
    if gravity.ndim != 2 or gravity.shape != (angles.shape[0], 3):
        raise ValueError("gravity must have shape [batch, 3]")
    gravity = gravity.to(dtype=dtype)
    norms = gravity.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(gravity)
    fallback[:, 2] = 1.0
    directions = torch.where(norms > 1e-8, gravity / norms.clamp_min(1e-8), fallback)

    random_vectors = torch.randn_like(directions)
    axes = random_vectors - (random_vectors * directions).sum(dim=-1, keepdim=True) * directions
    axis_norms = axes.norm(dim=-1, keepdim=True)

    basis_index = directions.abs().argmin(dim=-1)
    basis = torch.nn.functional.one_hot(basis_index, num_classes=3).to(dtype=dtype)
    deterministic_axes = torch.linalg.cross(directions, basis, dim=-1)
    deterministic_axes = deterministic_axes / deterministic_axes.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    axes = torch.where(axis_norms > 1e-8, axes / axis_norms.clamp_min(1e-8), deterministic_axes)

    x, y, z = axes.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack((zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=-1).reshape(-1, 3, 3)
    identity = torch.eye(3, device=angles.device, dtype=dtype).expand(len(angles), -1, -1)
    outer = axes.unsqueeze(-1) * axes.unsqueeze(-2)
    cosine = angles.to(dtype=dtype).cos().view(-1, 1, 1)
    sine = angles.to(dtype=dtype).sin().view(-1, 1, 1)
    return cosine * identity + (1.0 - cosine) * outer + sine * skew


def _sample_profile_components(
    profile_bank: dict[str, dict[str, Any]],
    limb: str,
    batch_size: int,
    mode: str,
    excluded_participants: Sequence[str] | None = None,
    participant_balanced: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one of the predefined joint or negative-control profile schemes."""
    if mode not in PROFILE_SAMPLING_MODES:
        raise ValueError(
            f"Unknown placement profile sampling mode '{mode}'; "
            f"expected one of {PROFILE_SAMPLING_MODES}"
        )
    profile = profile_bank[limb]
    device = profile["rotation"].device

    def sample_indices(source: dict[str, Any], count: int) -> torch.Tensor:
        size = int(source["rotation"].shape[0])
        names = tuple(str(value) for value in source.get("participant", ()))
        if participant_balanced:
            if len(names) != size or not all(names):
                raise ValueError(
                    "Participant-balanced placement sampling requires complete identities"
                )
            unique_names = tuple(sorted(set(names)))
            membership = torch.as_tensor(
                [[name == group for name in names] for group in unique_names],
                device=device,
                dtype=torch.float32,
            )
            allowed_groups = torch.ones(
                (count, len(unique_names)), device=device, dtype=torch.float32
            )
            if excluded_participants is not None:
                if len(excluded_participants) != count:
                    raise ValueError(
                        "excluded_participants must contain one identity per batch sample"
                    )
                allowed_groups = torch.as_tensor(
                    [
                        [group != str(excluded) for group in unique_names]
                        for excluded in excluded_participants
                    ],
                    device=device,
                    dtype=torch.float32,
                )
                if bool((allowed_groups.sum(dim=1) <= 0).any().item()):
                    raise ValueError("Participant exclusion removed every placement profile")
            group_indices = torch.multinomial(allowed_groups, 1).squeeze(1)
            record_weights = membership[group_indices]
            return torch.multinomial(record_weights, 1).squeeze(1)
        if excluded_participants is None:
            return torch.randint(size, (count,), device=device)
        if len(excluded_participants) != count:
            raise ValueError("excluded_participants must contain one identity per batch sample")
        if len(names) != size:
            raise ValueError("Placement profile participant metadata is incomplete")
        allowed = torch.as_tensor(
            [
                [not name or name != str(excluded) for name in names]
                for excluded in excluded_participants
            ],
            device=device,
            dtype=torch.float32,
        )
        if bool((allowed.sum(dim=1) <= 0).any().item()):
            raise ValueError("Participant exclusion removed every placement profile")
        return torch.multinomial(allowed, 1).squeeze(1)

    if mode == "joint":
        indices = sample_indices(profile, batch_size)
        return (
            profile["rotation"][indices],
            profile["gain"][indices],
            profile["noise_std"][indices],
            profile["angle_radians"][indices],
        )
    marginal_modes = {
        "matched_random",
        "global_matched_random",
        "gravity_matched_random",
        "global_gravity_matched_random",
    }
    if mode in marginal_modes:
        if mode in {"global_matched_random", "global_gravity_matched_random"}:
            profile = {
                field: torch.cat([profile_bank[key][field] for key in LIMB_KEYS], dim=0)
                for field in ("rotation", "angle_radians", "gain", "noise_std")
            }
            profile["participant"] = tuple(
                name for key in LIMB_KEYS for name in profile_bank[key]["participant"]
            )
        angle_indices = sample_indices(profile, batch_size)
        gain_indices = sample_indices(profile, batch_size)
        noise_indices = sample_indices(profile, batch_size)
        sampled_angles = profile["angle_radians"][angle_indices]
        if mode in {"gravity_matched_random", "global_gravity_matched_random"}:
            rotations = torch.eye(
                3,
                device=sampled_angles.device,
                dtype=profile["rotation"].dtype,
            ).expand(batch_size, -1, -1)
        else:
            rotations = _random_axis_rotations(sampled_angles, profile["rotation"].dtype)
        return (
            rotations,
            profile["gain"][gain_indices],
            profile["noise_std"][noise_indices],
            sampled_angles,
        )

    wrong_limb_profiles = [profile_bank[key] for key in LIMB_KEYS if key != limb]
    rotations = torch.cat([item["rotation"] for item in wrong_limb_profiles], dim=0)
    gains = torch.cat([item["gain"] for item in wrong_limb_profiles], dim=0)
    noise = torch.cat([item["noise_std"] for item in wrong_limb_profiles], dim=0)
    angles = torch.cat([item["angle_radians"] for item in wrong_limb_profiles], dim=0)
    wrong_profile = {
        "rotation": rotations,
        "gain": gains,
        "noise_std": noise,
        "angle_radians": angles,
        "participant": tuple(name for item in wrong_limb_profiles for name in item["participant"]),
    }
    indices = sample_indices(wrong_profile, batch_size)
    return rotations[indices], gains[indices], noise[indices], angles[indices]


def apply_calibrated_one_side_per_pair(
    batch: dict[str, Any],
    profile_bank: dict[str, dict[str, Any]],
    probability: float,
    *,
    use_rotation: bool = True,
    use_gain: bool = True,
    use_noise: bool = True,
    sampling_mode: str = "joint",
    exclude_source_participant: bool = False,
    participant_balanced: bool = False,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Sample an empirical training-participant placement profile for one side per pair."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("calibrated augmentation probability must lie in [0, 1]")
    if not any((use_rotation, use_gain, use_noise)):
        raise ValueError("At least one calibrated placement component must be enabled")
    if sampling_mode not in PROFILE_SAMPLING_MODES:
        raise ValueError(
            f"Unknown placement profile sampling mode '{sampling_mode}'; "
            f"expected one of {PROFILE_SAMPLING_MODES}"
        )
    if "left_wrist" in batch:
        pairs = LIMB_PAIRS
        profile_keys = {
            "left_wrist": "left_wrist",
            "right_wrist": "right_wrist",
            "left_ankle": "left_ankle",
            "right_ankle": "right_ankle",
        }
    elif "left" in batch and "right" in batch:
        pairs = {"arm": ("left", "right")}
        # The frozen REALDISP profile describes wrist-worn sensors.  MP1 uses
        # the same bilateral upper-arm role, so retain those profile marginals
        # while adapting only the batch field names.
        profile_keys = {"left": "left_wrist", "right": "right_wrist"}
    else:
        raise ValueError("Calibrated placement augmentation requires a bilateral sensor pair")
    augmented = dict(batch)
    targets: dict[str, torch.Tensor] = {}
    excluded_participants = None
    if exclude_source_participant:
        participants = batch.get("participant")
        if participants is None:
            raise ValueError("Participant-exclusive sampling requires batch participant IDs")
        excluded_participants = tuple(str(value) for value in participants)
    for prefix, (left_key, right_key) in pairs.items():
        left, right = batch[left_key], batch[right_key]
        if left.shape != right.shape or left.ndim != 3 or left.shape[-1] != 3:
            raise ValueError(f"Invalid calibrated pair shapes: {left.shape}/{right.shape}")
        batch_size = left.shape[0]
        mask = torch.rand(batch_size, device=left.device) < probability
        choose_left = torch.rand(batch_size, device=left.device) < 0.5
        pair_severity = torch.zeros(batch_size, device=left.device, dtype=left.dtype)
        for key, values, side_mask in (
            (left_key, left, mask & choose_left),
            (right_key, right, mask & ~choose_left),
        ):
            rotation, gain, noise, severity = _sample_profile_components(
                profile_bank,
                profile_keys[key],
                batch_size,
                sampling_mode,
                excluded_participants=excluded_participants,
                participant_balanced=participant_balanced,
            )
            if use_rotation:
                if sampling_mode in {
                    "gravity_matched_random",
                    "global_gravity_matched_random",
                }:
                    rotation = _gravity_tilt_rotations(
                        severity,
                        values.mean(dim=1),
                        values.dtype,
                    )
                transformed = torch.einsum("bij,btj->bti", rotation, values)
            else:
                transformed = values
            if use_gain:
                transformed = gain.view(-1, 1, 1) * transformed
            if use_noise:
                transformed = transformed + noise.view(-1, 1, 1) * torch.randn_like(transformed)
            augmented[key] = torch.where(side_mask.view(-1, 1, 1), transformed, values)
            pair_severity = torch.where(side_mask, severity.to(left.dtype), pair_severity)
        augmented[f"{prefix}_rotated_left"] = mask & choose_left
        augmented[f"{prefix}_rotated_right"] = mask & ~choose_left
        targets[prefix] = 1.0 - mask.to(left.dtype)
        targets[f"{prefix}_severity_radians"] = pair_severity
        augmented[f"{prefix}_quality_target"] = targets[prefix]
    return augmented, targets
