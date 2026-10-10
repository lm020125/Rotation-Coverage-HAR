from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_targeted_spectral_expert_cv import (  # noqa: E402
    select_competitor_classes,
    supervised_contrastive_loss,
    targeted_focal_loss,
)


def test_competitors_use_bidirectional_focus_confusions() -> None:
    targets = np.array([17, 17, 17, 18, 18, 0, 16, 11, 4])
    predictions = np.array([0, 16, 17, 0, 11, 18, 17, 18, 4])
    selected = select_competitor_classes(
        predictions, targets, (17, 18), competitors_per_focus=2
    )
    assert selected == [0, 11, 16]


def test_targeted_losses_are_finite_and_differentiable() -> None:
    features = torch.randn(8, 16, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    contrastive = supervised_contrastive_loss(
        features, labels, labels < 3, temperature=0.1
    )
    logits = torch.randn(8, 5, requires_grad=True)
    focal = targeted_focal_loss(logits, labels, torch.ones(5), gamma=1.5)
    loss = contrastive + focal
    loss.backward()
    assert torch.isfinite(loss)
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
