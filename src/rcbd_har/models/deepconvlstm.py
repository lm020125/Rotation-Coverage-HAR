from __future__ import annotations

import torch
from torch import nn


class DeepConvLSTM(nn.Module):
    """DeepConvLSTM baseline for two synchronized wearable IMUs.

    This follows the published four-convolution/two-LSTM topology from
    Ordonez and Roggen (2016). Left and right 6-axis IMUs are concatenated only
    at the input so it remains a controlled baseline for bilateral RCBD.
    """

    def __init__(
        self,
        input_channels_per_sensor: int,
        num_classes: int,
        conv_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.5,
        num_sensors: int = 2,
    ) -> None:
        super().__init__()
        if num_sensors < 1:
            raise ValueError("num_sensors must be positive")
        self.num_sensors = int(num_sensors)
        input_channels = self.num_sensors * input_channels_per_sensor
        convolutions: list[nn.Module] = []
        for layer in range(4):
            convolutions.extend(
                (
                    nn.Conv1d(
                        input_channels if layer == 0 else conv_channels,
                        conv_channels,
                        kernel_size=5,
                    ),
                    nn.ReLU(inplace=True),
                )
            )
        self.convolutions = nn.Sequential(*convolutions)
        self.recurrent = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def _forward_sensors(self, *sensors: torch.Tensor) -> dict[str, torch.Tensor]:
        if len(sensors) != self.num_sensors:
            raise ValueError(
                f"Expected {self.num_sensors} sensor tensors, received {len(sensors)}"
            )
        reference_shape = sensors[0].shape
        if len(reference_shape) != 3 or any(
            sensor.shape != reference_shape for sensor in sensors[1:]
        ):
            shapes = [tuple(sensor.shape) for sensor in sensors]
            raise ValueError(f"Sensors must have equal [B,T,C] shapes, got {shapes}")
        values = torch.cat(sensors, dim=-1).transpose(1, 2)
        sequence = self.convolutions(values).transpose(1, 2)
        sequence, _ = self.recurrent(sequence)
        embedding = self.dropout(sequence[:, -1])
        return {"embedding": embedding, "logits": self.classifier(embedding)}

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        audio: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del audio
        return self._forward_sensors(left, right)


class FourLimbDeepConvLSTM(DeepConvLSTM):
    """Input-concatenation DeepConvLSTM using the four REALDISP limb sensors."""

    def __init__(
        self,
        input_channels_per_sensor: int,
        num_classes: int,
        conv_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__(
            input_channels_per_sensor=input_channels_per_sensor,
            num_classes=num_classes,
            conv_channels=conv_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
            num_sensors=4,
        )

    def forward(
        self,
        left_wrist: torch.Tensor,
        right_wrist: torch.Tensor,
        left_ankle: torch.Tensor,
        right_ankle: torch.Tensor,
        anchor: torch.Tensor | None = None,
        missing_fraction: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del anchor, missing_fraction
        return self._forward_sensors(
            left_wrist, right_wrist, left_ankle, right_ankle
        )
