from __future__ import annotations

import torch
from torch import nn


class ResidualTemporalBlock(nn.Module):
    def __init__(self, dim: int, dropout: float, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=padding, dilation=dilation)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        y = self.dropout(self.activation(self.norm1(y)))
        y = self.conv2(y.transpose(1, 2)).transpose(1, 2)
        y = self.dropout(self.activation(self.norm2(y)))
        return residual + y


class SharedIMUEncoder(nn.Module):
    """A shared temporal encoder for either wrist, accepting [B,T,C]."""

    def __init__(self, input_channels: int, hidden_dim: int, blocks: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            ResidualTemporalBlock(hidden_dim, dropout, dilation=2**i) for i in range(blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"IMU input must be [B,T,C], got {tuple(x.shape)}")
        x = self.projection(x)
        for block in self.blocks:
            x = block(x)
        return x


class TemporalAggregator(nn.Module):
    """Aggregate a feature sequence with a bidirectional GRU or a TCN mean pool."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, kind: str = "gru") -> None:
        super().__init__()
        self.kind = kind
        if kind == "gru":
            self.encoder = nn.GRU(
                input_dim,
                hidden_dim,
                batch_first=True,
                bidirectional=True,
                num_layers=1,
            )
            self.output_dim = hidden_dim * 2
        elif kind == "tcn":
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                ResidualTemporalBlock(hidden_dim, dropout, dilation=1),
                ResidualTemporalBlock(hidden_dim, dropout, dilation=2),
            )
            self.output_dim = hidden_dim
        else:
            raise ValueError(f"Unsupported temporal encoder: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "gru":
            x, _ = self.encoder(x)
        else:
            x = self.encoder(x)
        return x.mean(dim=1)


class MultiScaleTemporalAggregator(nn.Module):
    """Fuse centered or separate past/future crops at physical time scales."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        kind: str,
        scales_seconds: list[float],
        sample_rate: float,
        tokens_per_scale: int = 64,
        bidirectional_windows: bool = False,
    ) -> None:
        super().__init__()
        if not scales_seconds:
            raise ValueError("scales_seconds cannot be empty")
        self.scales_samples = [max(1, int(round(scale * sample_rate))) for scale in scales_seconds]
        self.tokens_per_scale = int(tokens_per_scale)
        self.bidirectional_windows = bool(bidirectional_windows)
        self.shared = TemporalAggregator(input_dim, hidden_dim, dropout, kind)
        output_dim = self.shared.output_dim
        self.views = 2 * len(scales_seconds) if self.bidirectional_windows else len(scales_seconds)
        if self.bidirectional_windows:
            self.view_embeddings = nn.Parameter(torch.zeros(self.views, output_dim))
        else:
            self.register_parameter("view_embeddings", None)
        self.gate = nn.Sequential(
            nn.Linear(output_dim * self.views, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.views),
        )
        if self.bidirectional_windows:
            self.direction_fusion = nn.Sequential(
                nn.Linear(output_dim * 4, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
            )
        self.output_dim = output_dim

    def _crop_and_pool(self, x: torch.Tensor, samples: int, direction: str) -> torch.Tensor:
        length = x.shape[1]
        crop_length = min(samples, length)
        center = length // 2
        if direction == "past":
            start = max(0, center - crop_length)
            crop_length = center - start
        elif direction == "future":
            start = center
            crop_length = min(crop_length, length - start)
        elif direction == "center":
            start = (length - crop_length) // 2
        else:
            raise ValueError(f"Unknown temporal direction: {direction}")
        if crop_length <= 0:
            raise ValueError(f"Input length {length} is too short for {direction} crop")
        crop = x[:, start : start + crop_length]
        tokens = min(self.tokens_per_scale, crop_length)
        return torch.nn.functional.adaptive_avg_pool1d(crop.transpose(1, 2), tokens).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.bidirectional_windows and x.shape[1] < 2 * max(self.scales_samples):
            required = 2 * max(self.scales_samples)
            raise ValueError(
                f"Past/future scales require at least {required} samples, got {x.shape[1]}"
            )
        directions = ("past", "future") if self.bidirectional_windows else ("center",)
        embeddings = [
            self.shared(self._crop_and_pool(x, samples, direction))
            for direction in directions
            for samples in self.scales_samples
        ]
        if self.view_embeddings is not None:
            embeddings = [
                value + self.view_embeddings[index] for index, value in enumerate(embeddings)
            ]
        joined = torch.cat(embeddings, dim=-1)
        gate_logits = self.gate(joined)
        stacked = torch.stack(embeddings, dim=1)
        if self.bidirectional_windows:
            weights = torch.softmax(
                gate_logits.view(gate_logits.shape[0], 2, len(self.scales_samples)), dim=-1
            )
            directional = (
                weights.unsqueeze(-1)
                * stacked.view(stacked.shape[0], 2, len(self.scales_samples), -1)
            ).sum(dim=2)
            past, future = directional[:, 0], directional[:, 1]
            fused = self.direction_fusion(
                torch.cat((past, future, past * future, (past - future).abs()), dim=-1)
            )
            return fused, weights
        weights = torch.softmax(gate_logits, dim=-1)
        return (weights.unsqueeze(-1) * stacked).sum(dim=1), weights


class AudioCNNEncoder(nn.Module):
    """Encode a precomputed log-mel spectrogram [B,M,F] into one embedding."""

    def __init__(self, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, output_dim))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 3:
            raise ValueError(f"Audio input must be [B,M,F], got {tuple(audio.shape)}")
        return self.head(self.features(audio.unsqueeze(1)))
