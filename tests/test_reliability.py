from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from rcbd_har.losses import symmetric_kl, symmetric_kl_per_sample
from rcbd_har.reliability import (
    load_reliability_table,
    relative_discriminative_retention,
    reliability_weighted_replacement_delta,
)


def test_symmetric_kl_per_sample_matches_existing_mean() -> None:
    first = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    second = torch.tensor([[1.0, 1.0], [0.5, 0.5]])
    values = symmetric_kl_per_sample(first, second)

    assert values.shape == (2,)
    assert torch.allclose(values.mean(), symmetric_kl(first, second))
    assert bool((values >= 0).all())


def test_relative_retention_uses_damage_beyond_clean_difficulty() -> None:
    target = torch.tensor([0, 0])
    clean = torch.tensor([[3.0, 0.0], [0.2, 0.0]])
    unchanged = clean.clone()
    damaged = torch.tensor([[0.0, 3.0], [0.0, 0.2]])

    safe = relative_discriminative_retention(clean, unchanged, target, 1.0)
    unsafe = relative_discriminative_retention(clean, damaged, target, 1.0)

    assert torch.allclose(safe, torch.ones_like(safe))
    assert bool((unsafe < safe).all())


def test_reliability_lookup_uses_worst_changed_pair_and_minimum(tmp_path) -> None:
    payload = {
        "protocol": {"final_test_accessed": False},
        "severity_bin_edges_degrees": [0.0, 30.0, 60.0, 180.0],
        "global_reliability": [[0.9, 0.8, 0.7], [0.85, 0.75, 0.65]],
        "class_reliability": [
            [[0.9, 0.8, 0.1], [0.85, 0.75, 0.6]],
            [[0.7, 0.6, 0.5], [0.65, 0.55, 0.45]],
        ],
    }
    path = tmp_path / "reliability.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    table = load_reliability_table(
        path,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_classes=2,
        minimum_weight=0.2,
    )
    labels = torch.tensor([0, 1])
    targets = {
        "arm_severity_radians": torch.deg2rad(torch.tensor([120.0, 45.0])),
        "leg_severity_radians": torch.deg2rad(torch.tensor([0.0, 20.0])),
    }

    weights = table.weights(labels, targets)

    assert torch.allclose(weights, torch.tensor([0.2, 0.6]))


def test_weighted_replacement_matches_baseline_at_one_and_clean_at_zero() -> None:
    clean = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
    augmented = torch.tensor([[1.0, 2.0], [0.5, 1.0]])
    target = torch.tensor([0, 1])
    class_weights = torch.tensor([1.0, 2.0])

    full = reliability_weighted_replacement_delta(
        clean, augmented, target, torch.ones(2), class_weights
    )
    expected = F.cross_entropy(augmented, target, weight=class_weights) - F.cross_entropy(
        clean, target, weight=class_weights
    )
    zero = reliability_weighted_replacement_delta(
        clean, augmented, target, torch.zeros(2), class_weights
    )

    assert torch.allclose(full, expected)
    assert torch.allclose(zero, torch.zeros_like(zero))
