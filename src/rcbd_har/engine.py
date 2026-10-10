from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import (
    orientation_information_loss,
    regularized_classification_loss,
    symmetric_kl,
    symmetric_kl_per_sample,
)
from .metrics import classification_metrics
from .models.orientation import proper_signed_permutation_matrices
from .placement import (
    apply_calibrated_one_side_per_pair,
    load_placement_profile_bank,
)
from .reliability import (
    load_reliability_table,
    reliability_weighted_replacement_delta,
)
from .utils import EarlyStopping, move_batch


def placement_profile_raw_scale(config: dict[str, Any]) -> float:
    """Return the raw-unit divisor used by calibrated placement noise.

    Placement profiles store noise in the raw accelerometer unit.  Both the
    fitted shared-RMS normalization and WEAR's fixed sensor-range scaling are
    global linear transforms, so the same profile remains well-defined after
    dividing its noise by the corresponding scale.
    """
    data = config.get("data", {})
    normalization = str(data.get("normalization", ""))
    if normalization == "train_rms_shared":
        try:
            scale = float(data["normalization_stats"]["shared"]["rms"][0])
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError(
                "Calibrated placement augmentation requires fitted shared RMS statistics"
            ) from error
    elif normalization == "scale_only":
        scale = float(data.get("normalization_scale", 0.0))
    else:
        raise ValueError(
            "Calibrated placement augmentation requires train_rms_shared or scale_only"
        )
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Calibrated placement augmentation requires a positive raw scale")
    return scale


def forward_batch(
    model: torch.nn.Module, batch: dict[str, Any], swap_pairs: bool = False
) -> dict[str, torch.Tensor]:
    """Dispatch either a bilateral or four-limb batch to its model."""
    if "left_wrist" in batch:
        left_wrist, right_wrist = batch["left_wrist"], batch["right_wrist"]
        left_ankle, right_ankle = batch["left_ankle"], batch["right_ankle"]
        if swap_pairs:
            left_wrist, right_wrist = right_wrist, left_wrist
            left_ankle, right_ankle = right_ankle, left_ankle
        missing_fraction = batch.get("missing_fraction")
        if swap_pairs and missing_fraction is not None:
            order = [1, 0, 3, 2, *([4] if missing_fraction.shape[1] == 5 else [])]
            missing_fraction = missing_fraction[:, order]
        return model(
            left_wrist,
            right_wrist,
            left_ankle,
            right_ankle,
            anchor=batch.get("anchor"),
            missing_fraction=missing_fraction,
        )
    left, right = batch["left"], batch["right"]
    if swap_pairs:
        left, right = right, left
    return model(left, right, batch.get("audio"))


def rotate_four_limb_batch(batch: dict[str, Any], probability: float) -> dict[str, Any]:
    """Apply independent proper axis rotations without mutating the input batch."""
    if "left_wrist" not in batch or probability <= 0:
        return batch
    if not 0.0 <= probability <= 1.0:
        raise ValueError("orientation augmentation probability must lie in [0, 1]")
    rotated = dict(batch)
    candidates = proper_signed_permutation_matrices().to(
        device=batch["left_wrist"].device, dtype=batch["left_wrist"].dtype
    )
    identity = torch.eye(3, device=candidates.device, dtype=candidates.dtype)
    identity_index = (candidates - identity).abs().sum(dim=(1, 2)).argmin()
    for key in ("left_wrist", "right_wrist", "left_ankle", "right_ankle"):
        values = batch[key]
        indices = torch.randint(len(candidates), (values.shape[0],), device=values.device)
        mask = torch.rand(values.shape[0], device=values.device) < probability
        indices = torch.where(mask, indices, identity_index)
        matrices = candidates[indices]
        transformed = torch.einsum("bij,btj->bti", matrices, values)
        rotated[key] = transformed
        rotated[f"{key}_rotation_index"] = indices
    return rotated


def _axis_angle_rotation_matrices(
    severity: torch.Tensor, max_degrees: float, dtype: torch.dtype
) -> torch.Tensor:
    """Create continuous SO(3) rotations with angles scaled by ``severity``."""
    batch_size = severity.shape[0]
    axes = torch.randn(batch_size, 3, device=severity.device, dtype=dtype)
    axes = axes / axes.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    angles = severity.to(dtype=dtype) * torch.deg2rad(
        torch.tensor(max_degrees, device=severity.device, dtype=dtype)
    )
    x, y, z = axes.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack((zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=-1).reshape(batch_size, 3, 3)
    identity = torch.eye(3, device=severity.device, dtype=dtype).expand(batch_size, -1, -1)
    outer = axes.unsqueeze(-1) * axes.unsqueeze(-2)
    cosine = angles.cos().view(-1, 1, 1)
    sine = angles.sin().view(-1, 1, 1)
    return cosine * identity + (1.0 - cosine) * outer + sine * skew


def rotate_one_side_per_pair(
    batch: dict[str, Any], probability: float, max_degrees: float
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Rotate one sensor in each bilateral pair and return pair-quality targets.

    The same continuous SO(3) rotation is applied to the whole window.  A target
    of one denotes an unchanged pair, while zero denotes the configured maximum
    rotation.  Inputs are not mutated.
    """
    if "left_wrist" in batch:
        pairs = {
            "arm": ("left_wrist", "right_wrist"),
            "leg": ("left_ankle", "right_ankle"),
        }
    elif "left" in batch and "right" in batch:
        pairs = {"arm": ("left", "right")}
    else:
        raise ValueError("Pair-quality rotation requires a bilateral sensor pair")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quality augmentation probability must lie in [0, 1]")
    if not 0.0 < max_degrees <= 180.0:
        raise ValueError("quality augmentation max_degrees must lie in (0, 180]")
    rotated = dict(batch)
    targets: dict[str, torch.Tensor] = {}
    for prefix, (left_key, right_key) in pairs.items():
        left, right = batch[left_key], batch[right_key]
        if left.shape != right.shape or left.ndim != 3 or left.shape[-1] != 3:
            raise ValueError(
                "Pair-quality rotation requires equal [B,T,3] pair tensors, "
                f"got {left.shape}/{right.shape}"
            )
        mask = torch.rand(left.shape[0], device=left.device) < probability
        severity = torch.rand(left.shape[0], device=left.device) * mask.to(left.dtype)
        matrices = _axis_angle_rotation_matrices(severity, max_degrees, left.dtype)
        rotate_left = torch.rand(left.shape[0], device=left.device) < 0.5
        transformed_left = torch.einsum("bij,btj->bti", matrices, left)
        transformed_right = torch.einsum("bij,btj->bti", matrices, right)
        left_mask = (mask & rotate_left).view(-1, 1, 1)
        right_mask = (mask & ~rotate_left).view(-1, 1, 1)
        rotated[left_key] = torch.where(left_mask, transformed_left, left)
        rotated[right_key] = torch.where(right_mask, transformed_right, right)
        targets[prefix] = 1.0 - severity
        rotated[f"{prefix}_quality_target"] = targets[prefix]
        rotated[f"{prefix}_rotated_left"] = mask & rotate_left
        rotated[f"{prefix}_rotated_right"] = mask & ~rotate_left
    return rotated, targets


def qmbd_quality_loss(
    clean_output: dict[str, torch.Tensor],
    augmented_output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Supervise clean and rotation-degraded QMBD pair qualities."""
    terms: list[torch.Tensor] = []
    for prefix in ("arm", "leg"):
        key = f"{prefix}_quality"
        if key not in clean_output or key not in augmented_output:
            raise ValueError("QMBD quality supervision requires arm/leg quality diagnostics")
        clean_quality = clean_output[key]
        augmented_quality = augmented_output[key]
        target = targets[prefix].to(augmented_quality).view(-1, 1, 1)
        target = target.expand_as(augmented_quality)
        terms.append(F.smooth_l1_loss(clean_quality, torch.ones_like(clean_quality)))
        terms.append(F.smooth_l1_loss(augmented_quality, target))
    return torch.stack(terms).mean()


def _masked_relative_factor_mse(
    clean: torch.Tensor,
    augmented: torch.Tensor,
    mask: torch.Tensor,
    *,
    stop_gradient: bool,
) -> torch.Tensor:
    """Scale-normalized factor error for samples that were actually perturbed."""
    if clean.shape != augmented.shape or clean.ndim < 2:
        raise ValueError(
            "Factor consistency requires equal batched tensors, "
            f"got {clean.shape}/{augmented.shape}"
        )
    if mask.shape != (clean.shape[0],):
        raise ValueError(
            f"Factor consistency mask must have shape [{clean.shape[0]}], got {mask.shape}"
        )
    target = clean.detach() if stop_gradient else clean
    reduce_dims = tuple(range(1, clean.ndim))
    squared_error = (augmented - target).square().mean(dim=reduce_dims)
    target_energy = target.square().mean(dim=reduce_dims).detach().clamp_min(1e-4)
    relative_error = squared_error / target_energy
    selected = mask.to(device=clean.device, dtype=torch.bool)
    if not bool(selected.any().item()):
        return augmented.sum() * 0.0
    return relative_error[selected].mean()


def rcbd_factor_consistency_loss(
    clean_output: dict[str, torch.Tensor],
    augmented_output: dict[str, torch.Tensor],
    augmented_batch: dict[str, Any],
    *,
    stop_gradient: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align RCBD common and canonical-difference factors after one-side rotation.

    Only the bilateral pair that was actually rotated contributes.  By default
    the clean factor is a stop-gradient target so the objective teaches the
    perturbed path without pulling the normal-placement representation toward it.
    """
    common_terms: list[torch.Tensor] = []
    canonical_terms: list[torch.Tensor] = []
    for prefix in ("arm", "leg"):
        common_key = f"{prefix}_common"
        canonical_key = f"{prefix}_canonical_difference"
        missing = []
        for key in (common_key, canonical_key):
            if key not in clean_output:
                missing.append(f"clean_output.{key}")
            if key not in augmented_output:
                missing.append(f"augmented_output.{key}")
        for key in (f"{prefix}_rotated_left", f"{prefix}_rotated_right"):
            if key not in augmented_batch:
                missing.append(f"augmented_batch.{key}")
        if missing:
            raise ValueError(
                f"FAPC requires RCBD factor diagnostics and rotation masks; missing={missing}"
            )
        mask = (
            augmented_batch[f"{prefix}_rotated_left"] | augmented_batch[f"{prefix}_rotated_right"]
        )
        common_terms.append(
            _masked_relative_factor_mse(
                clean_output[common_key],
                augmented_output[common_key],
                mask,
                stop_gradient=stop_gradient,
            )
        )
        canonical_terms.append(
            _masked_relative_factor_mse(
                clean_output[canonical_key],
                augmented_output[canonical_key],
                mask,
                stop_gradient=stop_gradient,
            )
        )
    return torch.stack(common_terms).mean(), torch.stack(canonical_terms).mean()


def combine_clean_anchored_gradients(
    clean_gradients: tuple[torch.Tensor | None, ...],
    robust_gradients: tuple[torch.Tensor | None, ...],
    *,
    epsilon: float = 1e-12,
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    """Combine two objectives while preventing robust gradients from opposing clean ones.

    The robust gradient is left unchanged when its global inner product with the
    clean gradient is non-negative.  Under a conflict it is projected onto the
    hyperplane orthogonal to the clean gradient before the two are added.  This
    asymmetric rule makes clean classification the anchor objective while still
    allowing robust-only parameters to learn.
    """
    if len(clean_gradients) != len(robust_gradients):
        raise ValueError("Clean and robust gradient lists must have equal length")
    paired = [
        (clean, robust)
        for clean, robust in zip(clean_gradients, robust_gradients)
        if clean is not None and robust is not None
    ]
    if paired:
        dot = torch.stack([(clean * robust).sum() for clean, robust in paired]).sum()
        clean_norm = torch.stack([clean.square().sum() for clean, _ in paired]).sum()
        robust_norm = torch.stack([robust.square().sum() for _, robust in paired]).sum()
        denominator = (clean_norm.sqrt() * robust_norm.sqrt()).clamp_min(epsilon)
        cosine = dot / denominator
        conflict = bool((dot < 0).item()) and bool((clean_norm > epsilon).item())
        coefficient = dot / clean_norm.clamp_min(epsilon) if conflict else dot.new_zeros(())
    else:
        reference = next(
            (value for value in (*clean_gradients, *robust_gradients) if value is not None),
            None,
        )
        if reference is None:
            return [None] * len(clean_gradients), {
                "gradient_cosine": 0.0,
                "gradient_conflict": 0.0,
                "gradient_projection_coefficient": 0.0,
            }
        coefficient = reference.new_zeros(())
        cosine = reference.new_zeros(())
        conflict = False

    combined: list[torch.Tensor | None] = []
    for clean, robust in zip(clean_gradients, robust_gradients):
        if clean is None:
            combined.append(robust)
            continue
        if robust is None:
            combined.append(clean)
            continue
        projected_robust = robust - coefficient * clean if conflict else robust
        combined.append(clean + projected_robust)
    return combined, {
        "gradient_cosine": float(cosine.detach()),
        "gradient_conflict": float(conflict),
        "gradient_projection_coefficient": float(coefficient.detach()),
    }


def orientation_equivariance_targets(
    selected: torch.Tensor, applied: torch.Tensor, rotations: torch.Tensor
) -> torch.Tensor:
    """Target R' satisfying R' @ Q = R after applying input rotation Q."""
    rotations = rotations.to(device=selected.device)
    original = rotations[selected]
    augmentation = rotations[applied]
    desired = torch.bmm(original, augmentation.transpose(1, 2))
    distances = (desired[:, None] - rotations[None]).abs().sum(dim=(2, 3))
    return distances.argmin(dim=1)


def compute_class_weights(
    loader: DataLoader, num_classes: int, device: torch.device
) -> torch.Tensor:
    labels = loader.dataset.frame["label"].to_numpy(dtype=int)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    """Return softmax probabilities in dataset row order without requiring labels."""
    model.eval()
    probabilities: list[np.ndarray] = []
    for raw_batch in loader:
        output = forward_batch(model, move_batch(raw_batch, device))
        probabilities.append(torch.softmax(output["logits"], dim=-1).cpu().numpy())
    if not probabilities:
        raise ValueError("Cannot predict an empty dataset")
    return np.concatenate(probabilities, axis=0)


def expand_probabilities_to_samples(probabilities: np.ndarray, frame: Any) -> dict[str, np.ndarray]:
    """Expand window probabilities to the nearest original sample per recording."""
    required = {"participant", "center", "num_samples"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Sample-wise expansion requires manifest columns: {sorted(missing)}")
    recording_column = "recording_id" if "recording_id" in frame.columns else "session"
    outputs: dict[str, np.ndarray] = {}
    for group_key, indices in frame.groupby(
        ["participant", recording_column], sort=False
    ).indices.items():
        indices = np.asarray(indices, dtype=int)
        centers = frame.iloc[indices]["center"].to_numpy(dtype=np.int64)
        order = np.argsort(centers, kind="stable")
        centers, indices = centers[order], indices[order]
        if len(np.unique(centers)) != len(centers):
            raise ValueError(f"Duplicate prediction centers in recording {group_key}")
        lengths = frame.iloc[indices]["num_samples"].unique()
        if len(lengths) != 1:
            raise ValueError(f"Inconsistent num_samples in recording {group_key}")
        sample_indices = np.arange(int(lengths[0]), dtype=np.int64)
        right = np.searchsorted(centers, sample_indices, side="left")
        right = np.clip(right, 0, len(centers) - 1)
        left = np.clip(right - 1, 0, len(centers) - 1)
        choose_right = np.abs(centers[right] - sample_indices) < np.abs(
            centers[left] - sample_indices
        )
        nearest = np.where(choose_right, right, left)
        name = "__".join(str(value) for value in group_key)
        outputs[name] = probabilities[indices[nearest]]
    return outputs


def samplewise_classification_metrics(
    probabilities: np.ndarray, frame: Any, num_classes: int
) -> dict[str, Any]:
    """Evaluate nearest-center window probabilities on the original sample timeline."""
    if "labels_path" not in frame.columns:
        raise ValueError("Sample-wise evaluation requires a labels_path manifest column")
    recording_column = "recording_id" if "recording_id" in frame.columns else "session"
    expanded = expand_probabilities_to_samples(probabilities, frame)
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    participant_targets: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    participant_predictions: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    for group_key, indices in frame.groupby(
        ["participant", recording_column], sort=False
    ).indices.items():
        name = "__".join(str(value) for value in group_key)
        rows = frame.iloc[np.asarray(indices, dtype=int)]
        label_paths = rows["labels_path"].drop_duplicates().tolist()
        if len(label_paths) != 1:
            raise ValueError(f"Recording {group_key} maps to multiple label arrays")
        truth = np.load(str(label_paths[0]), allow_pickle=False).astype(np.int64)
        predicted = expanded[name].argmax(axis=1).astype(np.int64)
        if len(truth) != len(predicted):
            raise ValueError(
                f"Sample count mismatch for {group_key}: {len(truth)} vs {len(predicted)}"
            )
        participant = str(group_key[0])
        targets.append(truth)
        predictions.append(predicted)
        participant_targets[participant].append(truth)
        participant_predictions[participant].append(predicted)
    metrics = classification_metrics(
        np.concatenate(targets), np.concatenate(predictions), num_classes
    )
    metrics["per_participant"] = {
        participant: classification_metrics(
            np.concatenate(participant_targets[participant]),
            np.concatenate(participant_predictions[participant]),
            num_classes,
        )
        for participant in sorted(participant_targets)
    }
    return metrics


def smooth_probabilities(
    probabilities: np.ndarray,
    frame: Any,
    radius_steps: int,
    sigma_steps: float,
) -> np.ndarray:
    """Gaussian smoothing inside each participant/session without boundary leakage."""
    if radius_steps <= 0:
        return probabilities
    offsets = np.arange(-radius_steps, radius_steps + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / max(sigma_steps, 1e-6)) ** 2)
    kernel /= kernel.sum()
    output = np.empty_like(probabilities)
    recording_column = (
        "recording_id"
        if "recording_id" in frame.columns
        else ("session" if "session" in frame.columns else None)
    )
    group_columns = ["participant"] + ([recording_column] if recording_column else [])
    grouped = frame.reset_index(drop=True).groupby(group_columns, sort=False).indices
    for indices in grouped.values():
        indices = np.asarray(indices, dtype=int)
        if "center" in frame.columns:
            centers = frame.iloc[indices]["center"].to_numpy(dtype=np.int64)
            order = np.argsort(centers, kind="stable")
            indices, centers = indices[order], centers[order]
            differences = np.diff(centers)
            if np.any(differences <= 0):
                raise ValueError(
                    "Smoothing requires unique, strictly increasing centers per recording"
                )
            positive = differences[differences > 0]
            expected = float(np.median(positive)) if len(positive) else 1.0
            split_points = np.flatnonzero(differences > 1.5 * expected) + 1
            segments = np.split(indices, split_points)
        else:
            segments = [indices]
        for segment in segments:
            group = probabilities[segment]
            padded = np.pad(group, ((radius_steps, radius_steps), (0, 0)), mode="edge")
            smoothed = np.stack(
                [
                    np.convolve(padded[:, cls], kernel, mode="valid")
                    for cls in range(group.shape[1])
                ],
                axis=-1,
            )
            output[segment] = smoothed
    return output


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    swap_stress: bool = False,
    smoothing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    targets: list[int] = []
    participants: list[str] = []
    logits_all: list[np.ndarray] = []
    swapped_logits_all: list[np.ndarray] = []
    maximum_logit_difference = 0.0
    # RCBD role tensors have shape [batch, time, hidden].  Keeping every tensor
    # until the end of evaluation can require more than 100 GB on WEAR.  These
    # diagnostics are simple means, so accumulate their sufficient statistics
    # online instead.
    role_statistics: defaultdict[str, dict[str, float]] = defaultdict(
        lambda: {
            "abs_sum": 0.0,
            "abs_count": 0.0,
            "switch_sum": 0.0,
            "switch_count": 0.0,
        }
    )
    quality_statistics: defaultdict[str, dict[str, float]] = defaultdict(
        lambda: {"sum": 0.0, "square_sum": 0.0, "count": 0.0}
    )
    orientation_confidence: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    orientation_selection: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    ordered_trust_statistics: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    swapped_trust_statistics: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        output = forward_batch(model, batch)
        targets.extend(batch["label"].cpu().tolist())
        participants.extend([str(p) for p in raw_batch["participant"]])
        logits_all.append(output["logits"].detach().cpu().numpy())
        for key, value in output.items():
            if key == "role_score" or key.endswith("_role_score"):
                roles = value.detach()
                statistics = role_statistics[key]
                statistics["abs_sum"] += float(roles.abs().sum().item())
                statistics["abs_count"] += float(roles.numel())
                if roles.ndim >= 2 and roles.shape[1] > 1:
                    switches = torch.diff(roles, dim=1).abs()
                    statistics["switch_sum"] += float(switches.sum().item())
                    statistics["switch_count"] += float(switches.numel())
            if key.endswith("_quality"):
                quality = value.detach().float()
                statistics = quality_statistics[key]
                statistics["sum"] += float(quality.sum().item())
                statistics["square_sum"] += float(quality.square().sum().item())
                statistics["count"] += float(quality.numel())
            if key.endswith("orientation_probabilities"):
                confidence = value.detach().max(dim=-1).values.cpu().numpy()
                orientation_confidence[key].append(confidence)
            if key.endswith("orientation_selected"):
                orientation_selection[key].append(value.detach().cpu().numpy())
            if key.endswith("_ordered_trust"):
                statistics = ordered_trust_statistics[key]
                statistics[0] += float(value.detach().sum().item())
                statistics[1] += float(value.numel())
        if swap_stress:
            swapped = forward_batch(model, batch, swap_pairs=True)
            swapped_logits_all.append(swapped["logits"].detach().cpu().numpy())
            maximum_logit_difference = max(
                maximum_logit_difference,
                (output["logits"] - swapped["logits"]).abs().max().item(),
            )
            for key, value in swapped.items():
                if key.endswith("_ordered_trust"):
                    statistics = swapped_trust_statistics[key]
                    statistics[0] += float(value.detach().sum().item())
                    statistics[1] += float(value.numel())

    logits_array = np.concatenate(logits_all, axis=0)
    probabilities = torch.softmax(torch.from_numpy(logits_array), dim=-1).numpy()
    raw_predictions = probabilities.argmax(axis=-1)
    raw_metrics = classification_metrics(targets, raw_predictions, num_classes)
    smoothing = smoothing or {}
    radius_steps = 0
    sigma_steps = 1.0
    if bool(smoothing.get("enabled", False)):
        stride = float(smoothing.get("stride_seconds", 0.5))
        radius_steps = int(round(float(smoothing.get("radius_seconds", 5.0)) / stride))
        sigma_steps = float(smoothing.get("sigma_seconds", 2.5)) / stride
        probabilities = smooth_probabilities(
            probabilities, loader.dataset.frame, radius_steps, sigma_steps
        )
    predictions = probabilities.argmax(axis=-1)
    metrics = classification_metrics(targets, predictions, num_classes)
    metrics["raw"] = {
        "accuracy": raw_metrics["accuracy"],
        "macro_f1": raw_metrics["macro_f1"],
        "weighted_f1": raw_metrics["weighted_f1"],
    }
    metrics["raw_confusion_matrix"] = raw_metrics["confusion_matrix"]
    metrics["participants"] = sorted(set(participants))
    target_array = np.asarray(targets)
    participant_array = np.asarray(participants)
    metrics["per_participant"] = {
        participant: classification_metrics(
            target_array[participant_array == participant],
            predictions[participant_array == participant],
            num_classes,
        )
        for participant in metrics["participants"]
    }
    if {
        "labels_path",
        "center",
        "num_samples",
    }.issubset(loader.dataset.frame.columns):
        metrics["samplewise"] = samplewise_classification_metrics(
            probabilities, loader.dataset.frame, num_classes
        )
    for key, statistics in role_statistics.items():
        prefix = key.removesuffix("_role_score") or "role"
        metrics[f"{prefix}_role_abs_mean"] = statistics["abs_sum"] / max(
            statistics["abs_count"], 1.0
        )
        metrics[f"{prefix}_role_switch_mean"] = statistics["switch_sum"] / max(
            statistics["switch_count"], 1.0
        )
    for key, statistics in quality_statistics.items():
        prefix = key.removesuffix("_quality")
        count = max(statistics["count"], 1.0)
        mean = statistics["sum"] / count
        variance = max(statistics["square_sum"] / count - mean * mean, 0.0)
        metrics[f"{prefix}_quality_mean"] = mean
        metrics[f"{prefix}_quality_std"] = float(np.sqrt(variance))
    for key, values in orientation_confidence.items():
        prefix = key.removesuffix("_orientation_probabilities")
        metrics[f"{prefix}_orientation_confidence"] = float(np.concatenate(values).mean())
    for key, values in orientation_selection.items():
        prefix = key.removesuffix("_orientation_selected")
        selected = np.concatenate(values).astype(int)
        counts = np.bincount(selected, minlength=24).astype(np.float64)
        distribution = counts / max(counts.sum(), 1.0)
        entropy = -(distribution * np.log(np.maximum(distribution, 1e-12))).sum()
        metrics[f"{prefix}_orientation_histogram"] = counts.astype(int).tolist()
        metrics[f"{prefix}_orientation_dominant_fraction"] = float(distribution.max())
        metrics[f"{prefix}_orientation_effective_count"] = float(np.exp(entropy))
    for key, statistics in ordered_trust_statistics.items():
        prefix = key.removesuffix("_ordered_trust")
        metrics[f"{prefix}_ordered_trust_mean"] = statistics[0] / max(statistics[1], 1.0)
    for key, statistics in swapped_trust_statistics.items():
        prefix = key.removesuffix("_ordered_trust")
        metrics[f"{prefix}_swapped_ordered_trust_mean"] = statistics[0] / max(statistics[1], 1.0)
    if swap_stress:
        swapped_logits = np.concatenate(swapped_logits_all, axis=0)
        swapped_probabilities = torch.softmax(torch.from_numpy(swapped_logits), dim=-1).numpy()
        if bool(smoothing.get("enabled", False)):
            swapped_probabilities = smooth_probabilities(
                swapped_probabilities, loader.dataset.frame, radius_steps, sigma_steps
            )
        swapped_predictions = swapped_probabilities.argmax(axis=-1)
        swapped_metrics = classification_metrics(targets, swapped_predictions, num_classes)
        metrics["swap_stress"] = {
            "macro_f1": swapped_metrics["macro_f1"],
            "macro_f1_delta": float(swapped_metrics["macro_f1"] - metrics["macro_f1"]),
            "prediction_agreement": float(np.mean(predictions == swapped_predictions)),
            "maximum_logit_difference": maximum_logit_difference,
        }
    return metrics


def validation_selection_score(metrics: dict[str, Any], mode: str = "macro_f1") -> float:
    """Return the validation objective used for checkpointing and early stopping."""
    clean = float(metrics["macro_f1"])
    mode = str(mode).lower()
    if mode == "macro_f1":
        return clean
    if "swap_stress" not in metrics:
        raise ValueError(f"Validation selection mode '{mode}' requires swap-stress metrics")
    swapped = float(metrics["swap_stress"]["macro_f1"])
    if mode == "worst_case_macro_f1":
        return min(clean, swapped)
    if mode == "harmonic_macro_f1":
        return 2.0 * clean * swapped / max(clean + swapped, 1e-12)
    raise ValueError(f"Unknown validation selection mode: {mode}")


@torch.no_grad()
def evaluate_pair_rotation_stress(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    *,
    max_degrees: float = 90.0,
    seed: int = 0,
    smoothing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic one-side-per-pair SO(3) placement stress."""
    model.eval()
    targets: list[int] = []
    participants: list[str] = []
    logits_all: list[np.ndarray] = []
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            rotated_batch, _ = rotate_one_side_per_pair(
                batch, probability=1.0, max_degrees=max_degrees
            )
            output = forward_batch(model, rotated_batch)
            logits_all.append(output["logits"].detach().cpu().numpy())
            targets.extend(batch["label"].detach().cpu().tolist())
            participants.extend(str(value) for value in raw_batch["participant"])
    if not logits_all:
        raise ValueError("Cannot evaluate placement stress on an empty dataset")
    logits = np.concatenate(logits_all, axis=0)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    smoothing = smoothing or {}
    if bool(smoothing.get("enabled", False)):
        stride = float(smoothing.get("stride_seconds", 0.5))
        radius = int(round(float(smoothing.get("radius_seconds", 5.0)) / stride))
        sigma = float(smoothing.get("sigma_seconds", 2.5)) / stride
        probabilities = smooth_probabilities(probabilities, loader.dataset.frame, radius, sigma)
    predictions = probabilities.argmax(axis=-1)
    metrics = classification_metrics(targets, predictions, num_classes)
    target_array = np.asarray(targets)
    participant_array = np.asarray(participants)
    metrics["participants"] = sorted(set(participants))
    metrics["per_participant"] = {
        participant: classification_metrics(
            target_array[participant_array == participant],
            predictions[participant_array == participant],
            num_classes,
        )
        for participant in metrics["participants"]
    }
    metrics["stress_seed"] = int(seed)
    metrics["max_degrees"] = float(max_degrees)
    metrics["rotation_probability"] = 1.0
    return metrics


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config: dict[str, Any],
    device: torch.device,
    output_dir: str | Path,
    fixed_epochs: int | None = None,
    checkpoint_name: str = "best.pt",
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    num_classes = int(config["data"]["num_classes"])
    optimizer_name = str(training.get("optimizer", "adamw")).lower()
    optimizer_class = {"adam": Adam, "adamw": AdamW}.get(optimizer_name)
    if optimizer_class is None:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    optimizer = optimizer_class(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    epochs = int(fixed_epochs if fixed_epochs is not None else training["epochs"])
    warmup_epochs = min(max(0, int(training.get("warmup_epochs", 0))), max(epochs - 1, 0))
    scheduler_name = str(training.get("scheduler", "cosine")).lower()
    if scheduler_name == "exponential":
        if warmup_epochs:
            raise ValueError("Exponential scheduler does not support warmup_epochs")
        scheduler = ExponentialLR(optimizer, gamma=float(training.get("learning_rate_decay", 0.9)))
    elif scheduler_name != "cosine":
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    elif warmup_epochs:
        warmup = LinearLR(
            optimizer,
            start_factor=1.0 / float(warmup_epochs),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1))
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    stopper = EarlyStopping(int(training.get("patience", 10)))
    selection_mode = str(training.get("selection_metric", "macro_f1")).lower()
    swap_validation = selection_mode != "macro_f1"
    class_weights = None
    if bool(training.get("class_weighting", True)):
        class_weights = compute_class_weights(train_loader, num_classes, device)
    checkpoint = output_dir / checkpoint_name
    history: list[dict[str, float]] = []
    accumulation_steps = max(1, int(training.get("gradient_accumulation_steps", 1)))
    max_nonfinite_batches = max(1, int(training.get("max_nonfinite_batches", 3)))
    conflict_aware = bool(training.get("conflict_aware_factor_consistency", False))
    if conflict_aware:
        if not float(training.get("quality_augmentation_probability", 0.0)):
            raise ValueError("Conflict-aware FAPC requires quality_augmentation_probability > 0")
        if float(training.get("orientation_augmentation_probability", 0.0)):
            raise ValueError("Conflict-aware FAPC cannot be combined with orientation augmentation")
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    placement_profile_path = training.get("quality_augmentation_profile_path")
    placement_profile_bank = None
    if placement_profile_path:
        shared_rms = placement_profile_raw_scale(config)
        placement_profile_bank = load_placement_profile_bank(
            placement_profile_path,
            device=device,
            dtype=trainable_parameters[0].dtype,
            shared_rms=shared_rms,
            required_limbs=(
                ("left_wrist", "right_wrist")
                if config["data"].get("limb_mode", "bilateral") == "bilateral"
                else ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
            ),
        )
    reliability_path = training.get("quality_augmentation_reliability_path")
    reliability_table = None
    if reliability_path:
        if placement_profile_bank is None:
            raise ValueError("Reliability-constrained augmentation requires a profile bank")
        if conflict_aware:
            raise ValueError("Reliability weighting is not implemented for conflict-aware FAPC")
        reliability_table = load_reliability_table(
            reliability_path,
            device=device,
            dtype=trainable_parameters[0].dtype,
            num_classes=num_classes,
            minimum_weight=float(training.get("quality_augmentation_minimum_reliability", 0.2)),
        )
    nonfinite_batches = 0
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        sums: defaultdict[str, float] = defaultdict(float)
        samples = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        optimizer.zero_grad(set_to_none=True)
        for step, raw_batch in enumerate(progress):
            batch = move_batch(raw_batch, device)
            output = forward_batch(model, batch)
            swapped_output = None
            if any(
                float(config["loss"].get(key, 0.0))
                for key in ("swap_consistency", "swap_classification", "selection_gate")
            ):
                swapped_output = forward_batch(model, batch, swap_pairs=True)
            loss, components = regularized_classification_loss(
                output, swapped_output, batch["label"], class_weights, config["loss"]
            )
            clean_anchor_loss = loss
            robust_loss: torch.Tensor | None = None
            quality_probability = float(training.get("quality_augmentation_probability", 0.0))
            if quality_probability and (
                "left_wrist" in batch or ("left" in batch and "right" in batch)
            ):
                if placement_profile_bank is not None:
                    augmented_batch, quality_targets = apply_calibrated_one_side_per_pair(
                        batch,
                        placement_profile_bank,
                        quality_probability,
                        use_rotation=bool(
                            training.get("quality_augmentation_profile_rotation", True)
                        ),
                        use_gain=bool(training.get("quality_augmentation_profile_gain", True)),
                        use_noise=bool(training.get("quality_augmentation_profile_noise", True)),
                        sampling_mode=str(
                            training.get("quality_augmentation_profile_sampling", "joint")
                        ),
                        exclude_source_participant=bool(
                            training.get("quality_augmentation_exclude_source_participant", False)
                        ),
                        participant_balanced=bool(
                            training.get("quality_augmentation_profile_participant_balanced", False)
                        ),
                    )
                else:
                    max_degrees = float(training.get("quality_augmentation_max_degrees", 90.0))
                    augmented_batch, quality_targets = rotate_one_side_per_pair(
                        batch, quality_probability, max_degrees
                    )
                augmented_output = forward_batch(model, augmented_batch)
                augmented_ce = F.cross_entropy(
                    augmented_output["logits"], batch["label"], weight=class_weights
                )
                clean_ce = F.cross_entropy(output["logits"], batch["label"], weight=class_weights)
                classification_mix = float(
                    training.get("quality_augmentation_classification_mix", 0.5)
                )
                if not 0.0 <= classification_mix <= 1.0:
                    raise ValueError("quality_augmentation_classification_mix must lie in [0, 1]")
                reliability = None
                if reliability_table is not None:
                    reliability = reliability_table.weights(batch["label"], quality_targets)
                    replacement_delta = reliability_weighted_replacement_delta(
                        output["logits"],
                        augmented_output["logits"],
                        batch["label"],
                        reliability,
                        class_weights,
                    )
                    loss = loss + classification_mix * replacement_delta
                    components["augmentation_reliability_mean"] = float(reliability.mean().detach())
                    components["augmentation_reliability_min"] = float(reliability.min().detach())
                elif conflict_aware:
                    robust_loss = classification_mix * augmented_ce
                else:
                    loss = loss + classification_mix * (augmented_ce - clean_ce)
                quality_weight = float(config["loss"].get("quality_supervision", 0.0))
                if quality_weight:
                    quality = qmbd_quality_loss(output, augmented_output, quality_targets)
                    if conflict_aware:
                        robust_loss = robust_loss + quality_weight * quality
                    else:
                        loss = loss + quality_weight * quality
                    components["quality_supervision"] = float(quality.detach())
                prediction_weight = float(
                    config["loss"].get("perturbation_prediction_consistency", 0.0)
                )
                if prediction_weight:
                    if reliability is None:
                        prediction_consistency = symmetric_kl(
                            output["logits"], augmented_output["logits"]
                        )
                    else:
                        prediction_consistency = (
                            reliability
                            * symmetric_kl_per_sample(output["logits"], augmented_output["logits"])
                        ).mean()
                    if conflict_aware:
                        robust_loss = robust_loss + prediction_weight * prediction_consistency
                    else:
                        loss = loss + prediction_weight * prediction_consistency
                    components["perturbation_prediction_consistency"] = float(
                        prediction_consistency.detach()
                    )
                common_weight = float(config["loss"].get("factor_common_consistency", 0.0))
                canonical_weight = float(config["loss"].get("factor_canonical_consistency", 0.0))
                if common_weight or canonical_weight:
                    common_consistency, canonical_consistency = rcbd_factor_consistency_loss(
                        output,
                        augmented_output,
                        augmented_batch,
                        stop_gradient=bool(
                            config["loss"].get("factor_consistency_stop_gradient", True)
                        ),
                    )
                    factor_objective = (
                        common_weight * common_consistency
                        + canonical_weight * canonical_consistency
                    )
                    if conflict_aware:
                        robust_loss = robust_loss + factor_objective
                    else:
                        loss = loss + factor_objective
                    components["factor_common_consistency"] = float(common_consistency.detach())
                    components["factor_canonical_consistency"] = float(
                        canonical_consistency.detach()
                    )
                components["quality_augmented_ce"] = float(augmented_ce.detach())
                if conflict_aware:
                    loss = clean_anchor_loss + robust_loss
                    components["clean_anchor"] = float(clean_anchor_loss.detach())
                    components["robust_objective"] = float(robust_loss.detach())
                components["total"] = float(loss.detach())
            orientation_probability = float(
                training.get("orientation_augmentation_probability", 0.0)
            )
            if (
                orientation_probability
                and "left_wrist" in batch
                and "arm_orientation_probabilities" in output
            ):
                rotated_batch = rotate_four_limb_batch(batch, orientation_probability)
                rotated_output = forward_batch(model, rotated_batch)
                prediction_consistency = symmetric_kl(output["logits"], rotated_output["logits"])
                embedding_consistency = F.mse_loss(output["embedding"], rotated_output["embedding"])
                canonical_terms = []
                equivariance_terms = []
                rotations = proper_signed_permutation_matrices().to(device=device)
                for prefix, limb_keys in {
                    "arm": ("left_wrist", "right_wrist"),
                    "leg": ("left_ankle", "right_ankle"),
                }.items():
                    canonical_reference = output[f"{prefix}_orientation_canonical"]
                    canonical_error = F.mse_loss(
                        canonical_reference,
                        rotated_output[f"{prefix}_orientation_canonical"],
                    )
                    canonical_energy = canonical_reference.square().mean().detach().clamp_min(1e-6)
                    canonical_terms.append(canonical_error / canonical_energy)
                    applied = torch.cat(
                        [rotated_batch[f"{key}_rotation_index"] for key in limb_keys], dim=0
                    )
                    targets = orientation_equivariance_targets(
                        output[f"{prefix}_orientation_selected"].detach(), applied, rotations
                    )
                    equivariance_terms.append(
                        F.cross_entropy(rotated_output[f"{prefix}_orientation_logits"], targets)
                    )
                canonical_consistency = torch.stack(canonical_terms).mean()
                orientation_equivariance = torch.stack(equivariance_terms).mean()
                information = torch.stack(
                    [
                        orientation_information_loss(
                            rotated_output[f"{prefix}_orientation_probabilities"]
                        )
                        for prefix in ("arm", "leg")
                    ]
                ).mean()
                prediction_weight = float(
                    config["loss"].get("orientation_prediction_consistency", 0.0)
                )
                embedding_weight = float(
                    config["loss"].get("orientation_embedding_consistency", 0.0)
                )
                canonical_weight = float(
                    config["loss"].get("orientation_canonical_consistency", 0.0)
                )
                equivariance_weight = float(config["loss"].get("orientation_equivariance", 0.0))
                information_weight = float(config["loss"].get("orientation_information", 0.0))
                loss = (
                    loss
                    + prediction_weight * prediction_consistency
                    + embedding_weight * embedding_consistency
                    + canonical_weight * canonical_consistency
                    + equivariance_weight * orientation_equivariance
                    + information_weight * information
                )
                components["orientation_prediction_consistency"] = float(
                    prediction_consistency.detach()
                )
                components["orientation_embedding_consistency"] = float(
                    embedding_consistency.detach()
                )
                components["orientation_canonical_consistency"] = float(
                    canonical_consistency.detach()
                )
                components["orientation_equivariance"] = float(orientation_equivariance.detach())
                components["orientation_information"] = float(information.detach())
                components["total"] = float(loss.detach())
            if not bool(torch.isfinite(loss).item()):
                optimizer.zero_grad(set_to_none=True)
                nonfinite_batches += 1
                if nonfinite_batches >= max_nonfinite_batches:
                    raise FloatingPointError(
                        f"Encountered {nonfinite_batches} non-finite losses; aborting before "
                        "the optimizer can corrupt the model."
                    )
                continue
            if conflict_aware:
                if robust_loss is None:
                    raise RuntimeError("Conflict-aware FAPC did not construct a robust objective")
                clean_gradients = torch.autograd.grad(
                    clean_anchor_loss,
                    trainable_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                robust_gradients = torch.autograd.grad(
                    robust_loss,
                    trainable_parameters,
                    allow_unused=True,
                )
                combined_gradients, gradient_diagnostics = combine_clean_anchored_gradients(
                    clean_gradients, robust_gradients
                )
                for parameter, gradient in zip(trainable_parameters, combined_gradients):
                    if gradient is None:
                        continue
                    gradient = gradient / accumulation_steps
                    if parameter.grad is None:
                        parameter.grad = gradient.detach()
                    else:
                        parameter.grad.add_(gradient.detach())
                components.update(gradient_diagnostics)
            else:
                (loss / accumulation_steps).backward()
            if (step + 1) % accumulation_steps == 0 or step + 1 == len(train_loader):
                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=5.0, error_if_nonfinite=True
                    )
                except RuntimeError as exc:
                    optimizer.zero_grad(set_to_none=True)
                    nonfinite_batches += 1
                    if nonfinite_batches >= max_nonfinite_batches:
                        raise FloatingPointError(
                            f"Encountered {nonfinite_batches} non-finite gradient updates; "
                            "aborting before the optimizer can corrupt the model."
                        ) from exc
                    continue
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            batch_size = int(batch["label"].shape[0])
            samples += batch_size
            for name, value in components.items():
                sums[name] += value * batch_size
            progress.set_postfix(loss=f"{components['total']:.4f}")
        scheduler.step()
        val_metrics = (
            evaluate(
                model,
                val_loader,
                device,
                num_classes,
                swap_stress=swap_validation,
                smoothing=config.get("evaluation"),
            )
            if val_loader is not None
            else None
        )
        record = {
            "epoch": float(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "nonfinite_batches": float(nonfinite_batches),
        }
        if val_metrics is not None:
            record["val_macro_f1"] = float(val_metrics["macro_f1"])
            selection_score = validation_selection_score(val_metrics, selection_mode)
            record["val_selection_score"] = selection_score
            if swap_validation:
                record["val_swap_macro_f1"] = float(val_metrics["swap_stress"]["macro_f1"])
        record.update({f"train_{key}": value / max(samples, 1) for key, value in sums.items()})
        history.append(record)
        should_save = fixed_epochs is not None and epoch == epochs
        if val_metrics is not None and fixed_epochs is None:
            should_save = stopper.update(selection_score)
        if should_save:
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "metrics": val_metrics or {},
                },
                checkpoint,
            )
        if fixed_epochs is None and stopper.should_stop:
            break
    if not checkpoint.exists():
        raise RuntimeError("Training ended without producing a checkpoint")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return model, history
