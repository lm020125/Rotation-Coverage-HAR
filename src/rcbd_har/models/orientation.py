from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
from torch import nn


def proper_signed_permutation_matrices() -> torch.Tensor:
    """Return the 24 axis-aligned proper rotations of a 3D coordinate frame."""
    matrices = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = torch.zeros(3, 3)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            if round(float(torch.det(matrix))) == 1:
                matrices.append(matrix)
    result = torch.stack(matrices)
    if result.shape != (24, 3, 3):
        raise RuntimeError(f"Expected 24 proper rotations, got {result.shape}")
    return result


@dataclass
class OrientationOutput:
    canonical: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor
    selected_index: torch.Tensor


class LatentOrientationCanonicalizer(nn.Module):
    """Infer a discrete sensor-axis orientation and map it to a canonical frame.

    The finite candidate set preserves axial information, unlike magnitude-only
    invariants. A deterministic straight-through argmax supplies a hard selected
    rotation while preserving softmax gradients during training.
    """

    def __init__(
        self, hidden_dim: int = 64, temperature: float = 1.0, mode: str = "legacy"
    ) -> None:
        super().__init__()
        self.register_buffer("rotations", proper_signed_permutation_matrices())
        self.temperature = float(temperature)
        self.mode = mode
        if mode not in {"legacy", "group_equivariant"}:
            raise ValueError(f"Unknown orientation mode: {mode}")
        output_dim = 1 if mode == "group_equivariant" else 24
        self.scorer = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _candidate_summaries(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = x.mean(dim=1), x.std(dim=1)
        minimum, maximum = x.amin(dim=1), x.amax(dim=1)
        rotations = self.rotations.to(dtype=x.dtype)
        positive = rotations.clamp_min(0)
        negative = (-rotations).clamp_min(0)
        rotated_mean = torch.einsum("kij,bj->bki", rotations, mean)
        rotated_std = torch.einsum("kij,bj->bki", rotations.abs(), std)
        rotated_min = torch.einsum("kij,bj->bki", positive, minimum) - torch.einsum(
            "kij,bj->bki", negative, maximum
        )
        rotated_max = torch.einsum("kij,bj->bki", positive, maximum) - torch.einsum(
            "kij,bj->bki", negative, minimum
        )
        return torch.cat((rotated_mean, rotated_std, rotated_min, rotated_max), dim=-1)

    def forward(self, x: torch.Tensor) -> OrientationOutput:
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"Orientation canonicalizer expects [B,T,3], got {x.shape}")
        if self.mode == "group_equivariant":
            logits = self.scorer(self._candidate_summaries(x)).squeeze(-1)
        else:
            summary = torch.cat((x.mean(dim=1), x.std(dim=1), x.amin(dim=1), x.amax(dim=1)), dim=-1)
            logits = self.scorer(summary)
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        selected = logits.argmax(dim=-1)
        hard = torch.nn.functional.one_hot(selected, num_classes=24).to(dtype=x.dtype)
        weights = hard - probabilities.detach() + probabilities if self.training else hard
        mixed_rotation = torch.einsum("bk,kij->bij", weights, self.rotations.to(dtype=x.dtype))
        canonical = torch.einsum("bij,btj->bti", mixed_rotation, x)
        return OrientationOutput(canonical, logits, probabilities, weights.argmax(dim=-1))
