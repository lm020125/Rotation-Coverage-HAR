import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from rcbd_har.engine import (
    combine_clean_anchored_gradients,
    evaluate,
    expand_probabilities_to_samples,
    rcbd_factor_consistency_loss,
    rotate_one_side_per_pair,
    rotate_four_limb_batch,
    samplewise_classification_metrics,
    smooth_probabilities,
)
from rcbd_har.metrics import classification_metrics_from_confusion


def test_clean_anchored_gradients_project_only_conflicting_robust_component() -> None:
    clean = (torch.tensor([1.0, 0.0]),)
    robust = (torch.tensor([-2.0, 3.0]),)

    combined, diagnostics = combine_clean_anchored_gradients(clean, robust)

    assert torch.allclose(combined[0], torch.tensor([1.0, 3.0]))
    assert diagnostics["gradient_conflict"] == 1.0
    projected_robust = combined[0] - clean[0]
    assert torch.allclose((projected_robust * clean[0]).sum(), torch.tensor(0.0))


def test_clean_anchored_gradients_preserve_aligned_and_unshared_parameters() -> None:
    clean = (torch.tensor([1.0, 0.0]), None, torch.tensor([2.0]))
    robust = (torch.tensor([2.0, 1.0]), torch.tensor([4.0]), None)

    combined, diagnostics = combine_clean_anchored_gradients(clean, robust)

    assert torch.allclose(combined[0], torch.tensor([3.0, 1.0]))
    assert torch.allclose(combined[1], torch.tensor([4.0]))
    assert torch.allclose(combined[2], torch.tensor([2.0]))
    assert diagnostics["gradient_conflict"] == 0.0


def test_smoothing_does_not_cross_session_boundaries() -> None:
    probabilities = np.asarray([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3)
    frame = pd.DataFrame({"participant": ["P01"] * 6, "session": ["A"] * 3 + ["B"] * 3})
    smoothed = smooth_probabilities(probabilities, frame, radius_steps=2, sigma_steps=1.0)
    np.testing.assert_allclose(smoothed, probabilities)


def test_metrics_from_summed_confusion() -> None:
    metrics = classification_metrics_from_confusion([[3, 1], [1, 3]])
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] == 0.75


def test_axis_rotation_augmentation_preserves_vector_norms() -> None:
    batch = {
        key: torch.randn(5, 20, 3)
        for key in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
    }
    rotated = rotate_four_limb_batch(batch, probability=1.0)
    for key, values in batch.items():
        torch.testing.assert_close(values.norm(dim=-1), rotated[key].norm(dim=-1))


def test_pair_quality_rotation_changes_exactly_one_side_and_preserves_norms() -> None:
    torch.manual_seed(7)
    batch = {
        key: torch.randn(6, 20, 3)
        for key in ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
    }
    rotated, targets = rotate_one_side_per_pair(batch, probability=1.0, max_degrees=90.0)
    for prefix, (left_key, right_key) in {
        "arm": ("left_wrist", "right_wrist"),
        "leg": ("left_ankle", "right_ankle"),
    }.items():
        left_selected = rotated[f"{prefix}_rotated_left"]
        right_selected = rotated[f"{prefix}_rotated_right"]
        assert torch.all(left_selected ^ right_selected)
        assert torch.all((targets[prefix] >= 0) & (targets[prefix] <= 1))
        torch.testing.assert_close(
            batch[left_key].norm(dim=-1), rotated[left_key].norm(dim=-1), atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(
            batch[right_key].norm(dim=-1), rotated[right_key].norm(dim=-1), atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(rotated[left_key][right_selected], batch[left_key][right_selected])
        torch.testing.assert_close(rotated[right_key][left_selected], batch[right_key][left_selected])


def test_factor_consistency_uses_only_rotated_pairs_and_stops_clean_gradient() -> None:
    clean = {
        "arm_common": torch.ones(2, 3, 4, requires_grad=True),
        "arm_canonical_difference": torch.ones(2, 3, 4, requires_grad=True),
        "leg_common": torch.ones(2, 3, 4, requires_grad=True),
        "leg_canonical_difference": torch.ones(2, 3, 4, requires_grad=True),
    }
    augmented = {key: value.detach().clone().requires_grad_() for key, value in clean.items()}
    augmented["arm_common"].data[0].add_(1.0)
    augmented["arm_common"].data[1].add_(100.0)
    augmented["leg_canonical_difference"].data[1].add_(2.0)
    masks = {
        "arm_rotated_left": torch.tensor([True, False]),
        "arm_rotated_right": torch.tensor([False, False]),
        "leg_rotated_left": torch.tensor([False, False]),
        "leg_rotated_right": torch.tensor([False, True]),
    }
    common, canonical = rcbd_factor_consistency_loss(
        clean, augmented, masks, stop_gradient=True
    )
    torch.testing.assert_close(common, torch.tensor(0.5))
    torch.testing.assert_close(canonical, torch.tensor(2.0))
    (common + canonical).backward()
    assert all(value.grad is None for value in clean.values())
    assert augmented["arm_common"].grad is not None
    assert augmented["leg_canonical_difference"].grad is not None


def test_factor_consistency_is_zero_for_identical_factors() -> None:
    factors = {
        f"{prefix}_{name}": torch.randn(3, 5, 7)
        for prefix in ("arm", "leg")
        for name in ("common", "canonical_difference")
    }
    masks = {
        f"{prefix}_rotated_{side}": torch.ones(3, dtype=torch.bool)
        for prefix in ("arm", "leg")
        for side in ("left", "right")
    }
    common, canonical = rcbd_factor_consistency_loss(factors, factors, masks)
    torch.testing.assert_close(common, torch.tensor(0.0))
    torch.testing.assert_close(canonical, torch.tensor(0.0))


def test_smoothing_sorts_centers_and_breaks_at_gaps() -> None:
    sorted_probabilities = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.6, 0.4]])
    sorted_frame = pd.DataFrame(
        {"participant": ["P"] * 3, "session": ["S"] * 3, "center": [0, 1, 2]}
    )
    expected = smooth_probabilities(sorted_probabilities, sorted_frame, 1, 1.0)
    order = [2, 0, 1]
    shuffled = smooth_probabilities(
        sorted_probabilities[order], sorted_frame.iloc[order].reset_index(drop=True), 1, 1.0
    )
    np.testing.assert_allclose(shuffled, expected[order])
    gap_probabilities = np.asarray([[1.0, 0.0]] * 2 + [[0.0, 1.0]] * 2)
    gap_frame = pd.DataFrame(
        {"participant": ["P"] * 4, "session": ["S"] * 4, "center": [0, 1, 100, 101]}
    )
    np.testing.assert_allclose(
        smooth_probabilities(gap_probabilities, gap_frame, 2, 1.0), gap_probabilities
    )


def test_samplewise_expansion_uses_nearest_center_and_earlier_ties() -> None:
    probabilities = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    frame = pd.DataFrame(
        {
            "participant": ["P", "P"],
            "recording_id": ["R", "R"],
            "center": [0, 2],
            "num_samples": [4, 4],
        }
    )
    expanded = expand_probabilities_to_samples(probabilities, frame)["P__R"]
    np.testing.assert_array_equal(expanded.argmax(axis=-1), [0, 0, 1, 1])


def test_samplewise_metrics_use_original_labels_and_report_participants(tmp_path) -> None:
    labels_a = tmp_path / "labels_a.npy"
    labels_b = tmp_path / "labels_b.npy"
    np.save(labels_a, np.asarray([0, 0, 1, 1]))
    np.save(labels_b, np.asarray([1, 1, 0, 0]))
    frame = pd.DataFrame(
        {
            "participant": ["P1", "P1", "P2", "P2"],
            "recording_id": ["R1", "R1", "R2", "R2"],
            "center": [0, 2, 0, 2],
            "num_samples": [4, 4, 4, 4],
            "labels_path": [labels_a, labels_a, labels_b, labels_b],
        }
    )
    probabilities = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
    )
    metrics = samplewise_classification_metrics(probabilities, frame, num_classes=2)
    assert metrics["macro_f1"] == 1.0
    assert set(metrics["per_participant"]) == {"P1", "P2"}
    assert metrics["per_participant"]["P2"]["accuracy"] == 1.0


class _ParticipantDataset(Dataset):
    def __init__(self) -> None:
        self.frame = pd.DataFrame({"participant": ["P1", "P1", "P2", "P2"]})

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, object]:
        label = index % 2
        signal = torch.tensor([[float(label)]])
        return {
            "left": signal,
            "right": signal,
            "label": label,
            "participant": self.frame.iloc[index]["participant"],
        }


class _ParticipantModel(torch.nn.Module):
    def forward(
        self, left: torch.Tensor, right: torch.Tensor, audio: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        labels = left[:, 0, 0].long()
        return {"logits": torch.nn.functional.one_hot(labels, num_classes=2).float() * 5.0}


class _RoleDiagnosticModel(_ParticipantModel):
    def forward(
        self, left: torch.Tensor, right: torch.Tensor, audio: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        output = super().forward(left, right, audio)
        base = left[:, 0, 0]
        output["arm_role_score"] = torch.stack(
            [base, -base, 2.0 * base], dim=1
        ).unsqueeze(-1)
        return output


class _TrustDiagnosticModel(_ParticipantModel):
    def forward(
        self, left: torch.Tensor, right: torch.Tensor, audio: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        output = super().forward(left, right, audio)
        output["arm_ordered_trust"] = torch.full(
            (left.shape[0], left.shape[1], 1), 0.75, device=left.device
        )
        return output


def test_evaluate_reports_metrics_for_each_participant() -> None:
    metrics = evaluate(
        _ParticipantModel(),
        DataLoader(_ParticipantDataset(), batch_size=2, shuffle=False),
        torch.device("cpu"),
        num_classes=2,
    )
    assert set(metrics["per_participant"]) == {"P1", "P2"}
    assert metrics["per_participant"]["P1"]["accuracy"] == 1.0
    assert metrics["per_participant"]["P2"]["macro_f1"] == 1.0


def test_evaluate_streams_role_diagnostics_across_batches() -> None:
    metrics = evaluate(
        _RoleDiagnosticModel(),
        DataLoader(_ParticipantDataset(), batch_size=2, shuffle=False),
        torch.device("cpu"),
        num_classes=2,
    )
    # Two samples have base=0 and two have base=1.  For base=1 the absolute
    # sequence is [1, 1, 2] and consecutive absolute changes are [2, 3].
    assert metrics["arm_role_abs_mean"] == 4.0 / 6.0
    assert metrics["arm_role_switch_mean"] == 5.0 / 4.0


def test_evaluate_streams_clean_and_swapped_trust_diagnostics() -> None:
    metrics = evaluate(
        _TrustDiagnosticModel(),
        DataLoader(_ParticipantDataset(), batch_size=2, shuffle=False),
        torch.device("cpu"),
        num_classes=2,
        swap_stress=True,
    )
    assert metrics["arm_ordered_trust_mean"] == 0.75
    assert metrics["arm_swapped_ordered_trust_mean"] == 0.75
