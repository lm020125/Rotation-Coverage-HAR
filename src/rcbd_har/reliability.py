from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


PAIR_NAMES = ("arm", "leg")


def relative_discriminative_retention(
    clean_logits: torch.Tensor,
    augmented_logits: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Measure augmentation damage relative to each sample's clean difficulty.

    A value of one means that the perturbation did not increase cross-entropy.
    Increasing damage decays smoothly toward zero. The returned tensor is
    detached because the score is an offline teacher statistic, not a trainable
    escape route for the student.
    """
    if temperature <= 0:
        raise ValueError("Reliability temperature must be positive")
    if clean_logits.shape != augmented_logits.shape or clean_logits.ndim != 2:
        raise ValueError("Reliability logits must be equal [batch, classes] tensors")
    clean_ce = F.cross_entropy(clean_logits, target, reduction="none")
    augmented_ce = F.cross_entropy(augmented_logits, target, reduction="none")
    damage = (augmented_ce - clean_ce).clamp_min(0.0)
    return torch.exp(-damage / float(temperature)).detach()


@dataclass(frozen=True)
class FrozenReliabilityTable:
    bin_edges_degrees: torch.Tensor
    class_values: torch.Tensor
    global_values: torch.Tensor
    minimum_weight: float

    def weights(
        self,
        labels: torch.Tensor,
        quality_targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Look up a conservative sample weight from arm/leg severities."""
        if labels.ndim != 1:
            raise ValueError("Reliability labels must have shape [batch]")
        pair_weights: list[torch.Tensor] = []
        interior_edges = self.bin_edges_degrees[1:-1]
        for pair_index, pair in enumerate(PAIR_NAMES):
            key = f"{pair}_severity_radians"
            if key not in quality_targets:
                raise ValueError(f"Reliability weighting requires {key}")
            severity = quality_targets[key]
            degrees = torch.rad2deg(severity)
            bins = torch.bucketize(degrees, interior_edges, right=False)
            values = self.class_values[labels, pair_index, bins]
            unchanged = severity <= 1e-8
            pair_weights.append(torch.where(unchanged, torch.ones_like(values), values))
        weight = torch.stack(pair_weights, dim=1).amin(dim=1)
        return weight.clamp(min=self.minimum_weight, max=1.0).detach()


def load_reliability_table(
    path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_classes: int,
    minimum_weight: float,
) -> FrozenReliabilityTable:
    if not 0.0 <= minimum_weight <= 1.0:
        raise ValueError("minimum reliability weight must lie in [0, 1]")
    with Path(path).open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    protocol = payload.get("protocol", {})
    if protocol.get("final_test_accessed") is not False:
        raise ValueError("Reliability table must attest final_test_accessed=false")
    edges = torch.as_tensor(payload["severity_bin_edges_degrees"], device=device, dtype=dtype)
    if edges.ndim != 1 or len(edges) < 3 or not bool((edges[1:] > edges[:-1]).all()):
        raise ValueError("Reliability severity edges must be strictly increasing")
    bins = len(edges) - 1
    class_values = torch.as_tensor(payload["class_reliability"], device=device, dtype=dtype)
    global_values = torch.as_tensor(payload["global_reliability"], device=device, dtype=dtype)
    expected_class_shape = (num_classes, len(PAIR_NAMES), bins)
    expected_global_shape = (len(PAIR_NAMES), bins)
    if tuple(class_values.shape) != expected_class_shape:
        raise ValueError(
            f"Expected class reliability {expected_class_shape}, got {class_values.shape}"
        )
    if tuple(global_values.shape) != expected_global_shape:
        raise ValueError(
            f"Expected global reliability {expected_global_shape}, got {global_values.shape}"
        )
    if not bool(torch.isfinite(class_values).all() and torch.isfinite(global_values).all()):
        raise ValueError("Reliability table contains non-finite values")
    if bool(((class_values < 0) | (class_values > 1)).any()):
        raise ValueError("Class reliability values must lie in [0, 1]")
    return FrozenReliabilityTable(edges, class_values, global_values, minimum_weight)


def reliability_weighted_replacement_delta(
    clean_logits: torch.Tensor,
    augmented_logits: torch.Tensor,
    target: torch.Tensor,
    reliability: torch.Tensor,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Reliability-weighted clean-to-augmented CE replacement.

    With reliability=1 this exactly matches ``augmented_ce-clean_ce`` under
    PyTorch's weighted cross-entropy normalization. With reliability=0 the
    augmented view contributes no replacement, leaving the clean objective
    unchanged.
    """
    clean = F.cross_entropy(clean_logits, target, reduction="none")
    augmented = F.cross_entropy(augmented_logits, target, reduction="none")
    if class_weights is None:
        weights = torch.ones_like(clean)
    else:
        weights = class_weights[target]
    denominator = weights.sum().clamp_min(1e-8)
    return (reliability * weights * (augmented - clean)).sum() / denominator
