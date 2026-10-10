from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class VariableEmbedding(nn.Module):
    """Shared per-variable embedding adapted from the official DecomposeWHAR code."""

    def __init__(self, embedding_dim: int, patch_size: int, stride: int) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.projection = nn.Conv1d(1, embedding_dim, self.patch_size, self.stride)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # [batch, variables, time] -> [batch, variables, channels, patches]
        batch, variables, _ = values.shape
        flattened = values.reshape(batch * variables, 1, -1)
        flattened = F.pad(
            flattened,
            (0, self.patch_size - self.stride),
            mode="replicate",
        )
        embedded = self.projection(flattened)
        return embedded.reshape(batch, variables, embedded.shape[1], embedded.shape[2])


class DecomposeBlock(nn.Module):
    """Variable-preserving temporal extraction followed by channel/variable fusion."""

    def __init__(self, variables: int, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        width = variables * channels
        self.temporal = nn.Conv1d(
            width,
            width,
            kernel_size,
            padding="same",
            groups=width,
        )
        self.normalization = nn.BatchNorm1d(width)
        self.channel_fusion = nn.Sequential(
            nn.Conv1d(width, width, 1, groups=variables),
            nn.GELU(),
            nn.Conv1d(width, width, 1, groups=variables),
        )
        self.variable_fusion = nn.Sequential(
            nn.Conv1d(width, width, 1, groups=channels),
            nn.GELU(),
            nn.Conv1d(width, width, 1, groups=channels),
        )

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        batch, variables, channels, patches = embedded.shape
        values = embedded.reshape(batch, variables * channels, patches)
        values = self.channel_fusion(self.normalization(self.temporal(values)))
        values = values.reshape(batch, variables, channels, patches)
        values = values.permute(0, 2, 1, 3).reshape(
            batch, channels * variables, patches
        )
        values = self.variable_fusion(values)
        values = values.reshape(batch, channels, variables, patches).permute(0, 2, 1, 3)
        return embedded + values


class DecomposeCoordinationExpert(nn.Module):
    """Short-context raw-waveform expert for complementary full-body evidence.

    The decomposition stem follows DecomposeWHAR. A bidirectional recurrent temporal
    aggregator is used here so the module runs in the existing public PyTorch environment
    without a custom Mamba CUDA extension. Cross-sensor attention retains all four limb
    tokens instead of averaging upper/lower pair predictions.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        sensors: int = 4,
        variables: int = 3,
        embedding_dim: int = 32,
        patch_size: int = 10,
        stride: int = 5,
        blocks: int = 2,
        recurrent_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.sensors = int(sensors)
        self.variables = int(variables)
        self.embedding_dim = int(embedding_dim)
        self.embedding = VariableEmbedding(embedding_dim, patch_size, stride)
        self.blocks = nn.ModuleList(
            [DecomposeBlock(variables, embedding_dim) for _ in range(blocks)]
        )
        temporal_width = sensors * embedding_dim
        self.temporal = nn.GRU(
            temporal_width,
            recurrent_dim,
            batch_first=True,
            bidirectional=True,
        )
        sensor_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sensor_fusion = nn.TransformerEncoder(sensor_layer, num_layers=1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * recurrent_dim + embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * recurrent_dim + embedding_dim, num_classes),
        )
        self.representation_dim = 2 * recurrent_dim + embedding_dim

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[1] != self.sensors:
            raise ValueError("Expected raw input [batch, sensors, time, variables]")
        if values.shape[-1] != self.variables:
            raise ValueError("Raw input variable count does not match the model")
        batch, sensors, time, variables = values.shape
        per_sensor = values.reshape(batch * sensors, time, variables).permute(0, 2, 1)
        embedded = self.embedding(per_sensor)
        for block in self.blocks:
            embedded = block(embedded)
        embedded = embedded.mean(dim=1)
        patches = embedded.shape[-1]
        embedded = embedded.reshape(
            batch, sensors, self.embedding_dim, patches
        )
        temporal_tokens = embedded.permute(0, 3, 1, 2).reshape(
            batch, patches, sensors * self.embedding_dim
        )
        temporal_features, _ = self.temporal(temporal_tokens)
        temporal_summary = temporal_features.mean(dim=1)
        sensor_tokens = embedded.mean(dim=-1)
        sensor_summary = self.sensor_fusion(sensor_tokens).mean(dim=1)
        return torch.cat([temporal_summary, sensor_summary], dim=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(values))


class SpectralCoordinationEncoder(nn.Module):
    """Axis-invariant spectral power and cross-limb phase coordination encoder."""

    def __init__(self, sensors: int = 4, output_dim: int = 64) -> None:
        super().__init__()
        self.sensors = int(sensors)
        pair_count = sensors * (sensors - 1) // 2
        input_channels = sensors + 2 * pair_count
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, output_dim, 5, padding=2),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
            nn.Conv1d(output_dim, output_dim, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.output_dim = int(output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[1] != self.sensors:
            raise ValueError("Expected spectral input [batch, sensors, time, variables]")
        magnitude = torch.linalg.vector_norm(values.float(), dim=-1)
        magnitude = magnitude - magnitude.mean(dim=-1, keepdim=True)
        window = torch.hann_window(
            magnitude.shape[-1], device=magnitude.device, dtype=magnitude.dtype
        )
        spectrum = torch.fft.rfft(magnitude * window, dim=-1)[..., 1:]
        amplitude = torch.log1p(spectrum.abs())
        phase = spectrum / spectrum.abs().clamp_min(1e-6)
        phase_features = []
        for first in range(self.sensors):
            for second in range(first + 1, self.sensors):
                cross_phase = phase[:, first] * phase[:, second].conj()
                phase_features.extend([cross_phase.real, cross_phase.imag])
        features = torch.cat([amplitude, torch.stack(phase_features, dim=1)], dim=1)
        return self.network(features).squeeze(-1)


class TimeCoordinationEncoder(nn.Module):
    """Compact time-domain energy and zero-lag inter-limb correlation encoder."""

    def __init__(self, sensors: int = 4, output_dim: int = 32) -> None:
        super().__init__()
        self.sensors = int(sensors)
        pair_count = sensors * (sensors - 1) // 2
        input_dim = 3 * sensors + pair_count
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 2 * output_dim),
            nn.GELU(),
            nn.Linear(2 * output_dim, output_dim),
        )
        self.output_dim = int(output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[1] != self.sensors:
            raise ValueError("Expected coordination input [batch, sensors, time, variables]")
        magnitude = torch.linalg.vector_norm(values.float(), dim=-1)
        mean = magnitude.mean(dim=-1)
        standard_deviation = magnitude.std(dim=-1, unbiased=False)
        rms = magnitude.square().mean(dim=-1).sqrt()
        centered = magnitude - mean.unsqueeze(-1)
        normalized = centered / centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(
            1e-6
        )
        correlations = []
        for first in range(self.sensors):
            for second in range(first + 1, self.sensors):
                correlations.append((normalized[:, first] * normalized[:, second]).mean(dim=-1))
        statistics = torch.cat(
            [mean, standard_deviation, rms, torch.stack(correlations, dim=1)], dim=1
        )
        return self.network(statistics)


class TargetedSpectralCoordinationExpert(nn.Module):
    """Raw, spectral-phase, and time-coordination expert for hard confusions."""

    def __init__(
        self,
        num_classes: int,
        *,
        embedding_dim: int = 32,
        recurrent_dim: int = 128,
        spectral_dim: int = 64,
        coordination_dim: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.raw = DecomposeCoordinationExpert(
            num_classes,
            embedding_dim=embedding_dim,
            recurrent_dim=recurrent_dim,
            dropout=dropout,
        )
        self.spectral = SpectralCoordinationEncoder(output_dim=spectral_dim)
        self.coordination = TimeCoordinationEncoder(output_dim=coordination_dim)
        representation_dim = (
            self.raw.representation_dim + spectral_dim + coordination_dim
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(representation_dim),
            nn.Dropout(dropout),
            nn.Linear(representation_dim, num_classes),
        )
        self.representation_dim = representation_dim

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.raw.encode(values),
                self.spectral(values),
                self.coordination(values),
            ],
            dim=1,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(values))
