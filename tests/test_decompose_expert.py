from __future__ import annotations

import pytest
import torch

from rcbd_har.models.decompose_expert import (
    DecomposeCoordinationExpert,
    TargetedSpectralCoordinationExpert,
)


def test_decompose_coordination_expert_shape_and_gradients() -> None:
    model = DecomposeCoordinationExpert(19, embedding_dim=16, recurrent_dim=24)
    values = torch.randn(3, 4, 200, 3)
    logits = model(values)
    assert logits.shape == (3, 19)
    logits.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_decompose_coordination_expert_rejects_wrong_sensor_shape() -> None:
    model = DecomposeCoordinationExpert(19, embedding_dim=16, recurrent_dim=24)
    with pytest.raises(ValueError, match="sensors"):
        model(torch.randn(2, 2, 200, 3))


def test_targeted_spectral_coordination_expert_shape_and_gradients() -> None:
    model = TargetedSpectralCoordinationExpert(
        19,
        embedding_dim=16,
        recurrent_dim=24,
        spectral_dim=16,
        coordination_dim=8,
    )
    values = torch.randn(3, 4, 200, 3)
    representation = model.encode(values)
    logits = model(values)
    assert representation.shape == (3, model.representation_dim)
    assert logits.shape == (3, 19)
    logits.sum().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
