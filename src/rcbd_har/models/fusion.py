from __future__ import annotations

import torch
from torch import nn


class GatedMultimodalFusion(nn.Module):
    """Lightweight GMF over aligned modality embeddings of equal dimension."""

    def __init__(self, embedding_dim: int, modalities: int, dropout: float) -> None:
        super().__init__()
        self.modalities = modalities
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * modalities, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, modalities),
        )

    def forward(self, *embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if len(embeddings) != self.modalities:
            raise ValueError(f"Expected {self.modalities} modalities, got {len(embeddings)}")
        joined = torch.cat(embeddings, dim=-1)
        weights = torch.softmax(self.gate(joined), dim=-1)
        stacked = torch.stack(embeddings, dim=1)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        return fused, weights
