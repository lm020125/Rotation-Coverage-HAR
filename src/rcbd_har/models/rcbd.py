from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RCBDOutput:
    representation: torch.Tensor
    common: torch.Tensor
    raw_difference: torch.Tensor
    role_score: torch.Tensor
    canonical_difference: torch.Tensor


class RoleCanonicalBilateralDecomposition(nn.Module):
    """Swap-invariant bilateral decomposition that preserves role asymmetry.

    For shared-encoder wrist features h_l and h_r, the module computes
    s=(h_l+h_r)/2, d=(h_l-h_r)/2, r=tanh(g(h_l)-g(h_r)), and c=r*d.
    Under a left/right swap, both d and r change sign, so c remains unchanged.
    """

    def __init__(self, feature_dim: int, role_vector: bool = True) -> None:
        super().__init__()
        role_dim = feature_dim if role_vector else 1
        self.role_scorer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, role_dim),
        )
        self.feature_dim = feature_dim
        self.output_dim = feature_dim * 2

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> RCBDOutput:
        if left.shape != right.shape or left.ndim != 3:
            raise ValueError(
                f"RCBD inputs must be equal [B,T,D] tensors, got {left.shape}/{right.shape}"
            )
        common = 0.5 * (left + right)
        difference = 0.5 * (left - right)
        role = torch.tanh(self.role_scorer(left) - self.role_scorer(right))
        canonical = role * difference
        representation = torch.cat((common, canonical), dim=-1)
        return RCBDOutput(representation, common, difference, role, canonical)


class AnchorReferencedRoleCanonicalBilateralDecomposition(nn.Module):
    """Swap-invariant bilateral roles referenced to a shared body anchor.

    The anchor is projected into the limb feature space and removed from both
    sides before decomposition.  A shared scorer then assigns latent roles
    conditioned on that same anchor.  Swapping left/right negates both the
    residual difference and role score, leaving their product unchanged.
    """

    def __init__(self, feature_dim: int, role_vector: bool = True) -> None:
        super().__init__()
        role_dim = feature_dim if role_vector else 1
        self.anchor_reference = nn.Linear(feature_dim, feature_dim)
        self.role_scorer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, role_dim),
        )
        self.feature_dim = feature_dim
        self.output_dim = feature_dim * 2

    def forward(
        self, left: torch.Tensor, right: torch.Tensor, anchor: torch.Tensor
    ) -> RCBDOutput:
        if left.shape != right.shape or left.shape != anchor.shape or left.ndim != 3:
            raise ValueError(
                "Anchor-referenced RCBD inputs must be equal [B,T,D] tensors, "
                f"got {left.shape}/{right.shape}/{anchor.shape}"
            )
        reference = self.anchor_reference(anchor)
        left_residual = left - reference
        right_residual = right - reference
        common = 0.5 * (left_residual + right_residual)
        difference = 0.5 * (left_residual - right_residual)
        left_role = self.role_scorer(torch.cat((left_residual, anchor), dim=-1))
        right_role = self.role_scorer(torch.cat((right_residual, anchor), dim=-1))
        role = torch.tanh(left_role - right_role)
        canonical = role * difference
        representation = torch.cat((common, canonical), dim=-1)
        return RCBDOutput(representation, common, difference, role, canonical)
