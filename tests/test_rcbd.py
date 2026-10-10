import torch

from rcbd_har.engine import rotate_one_side_per_pair

from rcbd_har.losses import regularized_classification_loss
from rcbd_har.models.classifier import BilateralHARModel, MultiLimbHARModel, PairRepresentation
from rcbd_har.models.orientation import (
    LatentOrientationCanonicalizer,
    proper_signed_permutation_matrices,
)
from rcbd_har.models.rcbd import (
    AnchorReferencedRoleCanonicalBilateralDecomposition,
    RoleCanonicalBilateralDecomposition,
)
from rcbd_har.models.encoders import MultiScaleTemporalAggregator
from rcbd_har.engine import (
    orientation_equivariance_targets,
    qmbd_quality_loss,
    validation_selection_score,
)


def test_rcbd_is_exactly_swap_invariant() -> None:
    torch.manual_seed(3)
    layer = RoleCanonicalBilateralDecomposition(feature_dim=16).eval()
    left = torch.randn(4, 20, 16)
    right = torch.randn(4, 20, 16)
    forward = layer(left, right)
    swapped = layer(right, left)
    torch.testing.assert_close(forward.common, swapped.common, atol=0, rtol=0)
    torch.testing.assert_close(forward.role_score, -swapped.role_score, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(
        forward.canonical_difference, swapped.canonical_difference, atol=1e-7, rtol=1e-6
    )
    torch.testing.assert_close(forward.representation, swapped.representation, atol=1e-7, rtol=1e-6)


def test_complete_rcbd_model_is_swap_invariant_in_eval_mode() -> None:
    torch.manual_seed(5)
    model = BilateralHARModel(
        input_channels=6,
        num_classes=4,
        hidden_dim=24,
        encoder_blocks=2,
        dropout=0.1,
        variant="rcbd",
    ).eval()
    left = torch.randn(3, 50, 6)
    right = torch.randn(3, 50, 6)
    with torch.no_grad():
        logits = model(left, right)["logits"]
        swapped = model(right, left)["logits"]
    torch.testing.assert_close(logits, swapped, atol=1e-6, rtol=1e-5)


def test_controlled_decomposition_ablation_swap_properties() -> None:
    torch.manual_seed(5)
    left = torch.randn(3, 11, 8)
    right = torch.randn(3, 11, 8)

    ordered = PairRepresentation(hidden_dim=8, variant="ordered_diff", role_vector=True)
    ordered_forward, _ = ordered(left, right, "arm")
    ordered_swapped, _ = ordered(right, left, "arm")
    torch.testing.assert_close(ordered_forward[..., :8], ordered_swapped[..., :8])
    torch.testing.assert_close(ordered_forward[..., 8:], -ordered_swapped[..., 8:])

    no_role = PairRepresentation(hidden_dim=8, variant="absdiff", role_vector=True)
    no_role_forward, _ = no_role(left, right, "arm")
    no_role_swapped, _ = no_role(right, left, "arm")
    torch.testing.assert_close(no_role_forward, no_role_swapped, atol=0, rtol=0)


def test_qmbd_is_swap_invariant_and_has_exact_rcbd_limits() -> None:
    torch.manual_seed(11)
    left = torch.randn(3, 13, 8)
    right = torch.randn(3, 13, 8)

    fixed_one = PairRepresentation(
        hidden_dim=8, variant="qmbd", role_vector=True, quality_mode="fixed_one"
    ).eval()
    representation, diagnostics = fixed_one(left, right, "arm")
    assert fixed_one.rcbd is not None
    expected = fixed_one.rcbd(left, right).representation
    torch.testing.assert_close(representation, expected, atol=0, rtol=0)
    torch.testing.assert_close(diagnostics["arm_quality"], torch.ones(3, 13, 1))

    fixed_zero = PairRepresentation(
        hidden_dim=8, variant="qmbd", role_vector=True, quality_mode="fixed_zero"
    ).eval()
    zero_representation, zero_diagnostics = fixed_zero(left, right, "arm")
    torch.testing.assert_close(zero_representation[..., 8:], torch.zeros(3, 13, 8))
    torch.testing.assert_close(zero_diagnostics["arm_quality"], torch.zeros(3, 13, 1))

    learned = PairRepresentation(hidden_dim=8, variant="qmbd", role_vector=True).eval()
    learned_forward, forward_diagnostics = learned(left, right, "arm")
    learned_swapped, swapped_diagnostics = learned(right, left, "arm")
    torch.testing.assert_close(learned_forward, learned_swapped, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(
        forward_diagnostics["arm_quality"],
        swapped_diagnostics["arm_quality"],
        atol=1e-7,
        rtol=1e-6,
    )
    assert torch.all((forward_diagnostics["arm_quality"] > 0) & (forward_diagnostics["arm_quality"] < 1))


def test_qmbd_quality_supervision_backpropagates_into_quality_heads() -> None:
    torch.manual_seed(17)
    arm = PairRepresentation(hidden_dim=8, variant="qmbd", role_vector=True)
    leg = PairRepresentation(hidden_dim=8, variant="qmbd", role_vector=True)
    left = torch.randn(2, 10, 8)
    right = torch.randn(2, 10, 8)
    _, arm_clean = arm(left, right, "arm")
    _, leg_clean = leg(left, right, "leg")
    _, arm_augmented = arm(left + 0.25, right, "arm")
    _, leg_augmented = leg(left, right - 0.25, "leg")
    loss = qmbd_quality_loss(
        {**arm_clean, **leg_clean},
        {**arm_augmented, **leg_augmented},
        {"arm": torch.tensor([0.2, 0.8]), "leg": torch.tensor([0.4, 0.6])},
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert arm.quality_head is not None and leg.quality_head is not None
    assert any(parameter.grad is not None for parameter in arm.quality_head.parameters())
    assert any(parameter.grad is not None for parameter in leg.quality_head.parameters())


def test_trained_qmbd_pair_supports_runtime_quality_ablation() -> None:
    torch.manual_seed(29)
    pair = PairRepresentation(
        hidden_dim=8, variant="qmbd", role_vector=True, quality_mode="learned"
    )
    left = torch.randn(3, 13, 8)
    right = torch.randn(3, 13, 8)

    pair.quality_mode = "fixed_one"
    fixed_one, one_diagnostics = pair(left, right, "arm")
    pair.quality_mode = "fixed_zero"
    fixed_zero, zero_diagnostics = pair(left, right, "arm")

    torch.testing.assert_close(one_diagnostics["arm_quality"], torch.ones(3, 13, 1))
    torch.testing.assert_close(zero_diagnostics["arm_quality"], torch.zeros(3, 13, 1))
    torch.testing.assert_close(fixed_one[..., :8], fixed_zero[..., :8])
    assert torch.count_nonzero(fixed_one[..., 8:]) > 0
    torch.testing.assert_close(fixed_zero[..., 8:], torch.zeros_like(fixed_zero[..., 8:]))


def test_isolated_qmbd_separates_classification_and_quality_gradients() -> None:
    torch.manual_seed(31)
    pair = PairRepresentation(
        hidden_dim=8,
        variant="qmbd",
        role_vector=True,
        quality_gradient_mode="isolated",
    )
    assert pair.quality_head is not None
    left = torch.randn(2, 9, 8, requires_grad=True)
    right = torch.randn(2, 9, 8, requires_grad=True)
    representation, _ = pair(left, right, "arm")
    representation.square().mean().backward()

    assert left.grad is not None and right.grad is not None
    assert all(parameter.grad is None for parameter in pair.quality_head.parameters())

    pair.zero_grad(set_to_none=True)
    quality_left = torch.randn(2, 9, 8, requires_grad=True)
    quality_right = torch.randn(2, 9, 8, requires_grad=True)
    _, diagnostics = pair(quality_left, quality_right, "arm")
    diagnostics["arm_quality"].mean().backward()

    assert quality_left.grad is None and quality_right.grad is None
    assert any(parameter.grad is not None for parameter in pair.quality_head.parameters())


def test_rcbd_backpropagates_into_role_scorer() -> None:
    layer = RoleCanonicalBilateralDecomposition(feature_dim=8)
    left = torch.randn(2, 10, 8, requires_grad=True)
    right = torch.randn(2, 10, 8, requires_grad=True)
    layer(left, right).representation.square().mean().backward()
    assert left.grad is not None and torch.isfinite(left.grad).all()
    assert any(parameter.grad is not None for parameter in layer.role_scorer.parameters())


def test_anchor_referenced_rcbd_is_exactly_swap_invariant_and_uses_anchor() -> None:
    torch.manual_seed(13)
    layer = AnchorReferencedRoleCanonicalBilateralDecomposition(feature_dim=12).eval()
    left = torch.randn(3, 15, 12)
    right = torch.randn(3, 15, 12)
    anchor = torch.randn(3, 15, 12)
    forward = layer(left, right, anchor)
    swapped = layer(right, left, anchor)
    changed_anchor = layer(left, right, anchor + 0.5)
    torch.testing.assert_close(forward.role_score, -swapped.role_score, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(
        forward.canonical_difference, swapped.canonical_difference, atol=1e-7, rtol=1e-6
    )
    torch.testing.assert_close(forward.representation, swapped.representation, atol=1e-7, rtol=1e-6)
    assert not torch.allclose(forward.representation, changed_anchor.representation)


def test_multilimb_anchor_referenced_rcbd_is_independently_swap_invariant() -> None:
    torch.manual_seed(19)
    model = MultiLimbHARModel(
        input_channels=9,
        num_classes=6,
        hidden_dim=16,
        encoder_blocks=1,
        dropout=0.0,
        temporal="tcn",
        variant="rcbd",
        role_vector=True,
        sample_rate=25,
        scales_seconds=[0.5, 1, 2],
        tokens_per_scale=8,
        orientation_enabled=False,
        orientation_temperature=1.0,
        bidirectional_scales=True,
        anchor_mode="referenced",
    ).eval()
    limbs = [torch.randn(2, 125, 9) for _ in range(4)]
    anchor = torch.randn(2, 125, 9)
    missing = torch.zeros(2, 5)
    with torch.no_grad():
        reference = model(*limbs, anchor=anchor, missing_fraction=missing)["logits"]
        arm_swapped = model(
            limbs[1], limbs[0], limbs[2], limbs[3], anchor=anchor, missing_fraction=missing
        )["logits"]
        leg_swapped = model(
            limbs[0], limbs[1], limbs[3], limbs[2], anchor=anchor, missing_fraction=missing
        )["logits"]
    torch.testing.assert_close(reference, arm_swapped, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(reference, leg_swapped, atol=1e-6, rtol=1e-5)


def test_selective_rcbd_exposes_trainable_ordered_trust() -> None:
    torch.manual_seed(23)
    pair = PairRepresentation(
        hidden_dim=8,
        variant="rcbd",
        role_vector=True,
        selective=True,
    )
    left = torch.randn(2, 12, 8)
    right = torch.randn(2, 12, 8)
    anchor = torch.randn(2, 12, 8)
    representation, diagnostics = pair(left, right, "arm", anchor)
    assert representation.shape == (2, 12, 16)
    trust = diagnostics["arm_ordered_trust"]
    assert trust.shape == (2, 12, 1)
    assert torch.all((trust > 0) & (trust < 1))
    representation.square().mean().backward()
    assert any(parameter.grad is not None for parameter in pair.side_trust.parameters())


def test_selection_gate_supervision_prefers_clean_order_and_swapped_safety() -> None:
    logits = torch.zeros(2, 3)
    target = torch.tensor([0, 1])
    good = {
        "logits": logits,
        "arm_ordered_trust": torch.full((2, 4, 1), 0.9),
        "leg_ordered_trust": torch.full((2, 4, 1), 0.9),
    }
    good_swapped = {
        "logits": logits,
        "arm_ordered_trust": torch.full((2, 4, 1), 0.1),
        "leg_ordered_trust": torch.full((2, 4, 1), 0.1),
    }
    bad = {**good, "arm_ordered_trust": good_swapped["arm_ordered_trust"]}
    bad["leg_ordered_trust"] = good_swapped["leg_ordered_trust"]
    bad_swapped = {**good_swapped, "arm_ordered_trust": good["arm_ordered_trust"]}
    bad_swapped["leg_ordered_trust"] = good["leg_ordered_trust"]
    config = {"selection_gate": 1.0}
    good_loss, good_parts = regularized_classification_loss(
        good, good_swapped, target, None, config
    )
    bad_loss, _ = regularized_classification_loss(bad, bad_swapped, target, None, config)
    assert good_loss < bad_loss
    assert "selection_gate" in good_parts


def test_multilimb_selective_rcbd_forward_and_joint_swap_backward() -> None:
    torch.manual_seed(29)
    model = MultiLimbHARModel(
        input_channels=9,
        num_classes=6,
        hidden_dim=16,
        encoder_blocks=1,
        dropout=0.0,
        temporal="tcn",
        variant="rcbd",
        role_vector=True,
        sample_rate=25,
        scales_seconds=[0.5, 1, 2],
        tokens_per_scale=8,
        orientation_enabled=False,
        orientation_temperature=1.0,
        bidirectional_scales=True,
        anchor_mode="selective",
    )
    limbs = [torch.randn(2, 125, 9) for _ in range(4)]
    anchor = torch.randn(2, 125, 9)
    missing = torch.zeros(2, 5)
    output = model(*limbs, anchor=anchor, missing_fraction=missing)
    swapped = model(
        limbs[1], limbs[0], limbs[3], limbs[2], anchor=anchor, missing_fraction=missing
    )
    loss, parts = regularized_classification_loss(
        output,
        swapped,
        torch.tensor([0, 1]),
        None,
        {"swap_classification": 0.5, "selection_gate": 0.1},
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert {"swap_ce", "selection_gate"}.issubset(parts)
    assert output["arm_ordered_trust"].shape[0] == 2
    assert output["leg_ordered_trust"].shape[0] == 2


def test_multilimb_selective_dual_keeps_an_exactly_safe_expert() -> None:
    torch.manual_seed(31)
    model = MultiLimbHARModel(
        input_channels=9,
        num_classes=6,
        hidden_dim=16,
        encoder_blocks=1,
        dropout=0.0,
        temporal="tcn",
        variant="rcbd",
        role_vector=True,
        sample_rate=25,
        scales_seconds=[0.5, 1, 2],
        tokens_per_scale=8,
        orientation_enabled=False,
        orientation_temperature=1.0,
        bidirectional_scales=True,
        anchor_mode="selective_dual",
    ).eval()
    limbs = [torch.randn(2, 125, 9) for _ in range(4)]
    anchor = torch.randn(2, 125, 9)
    missing = torch.zeros(2, 5)
    with torch.no_grad():
        output = model(*limbs, anchor=anchor, missing_fraction=missing)
        swapped = model(
            limbs[1], limbs[0], limbs[3], limbs[2], anchor=anchor, missing_fraction=missing
        )

    assert output["logits"].shape == (2, 6)
    assert output["ordered_logits"].shape == (2, 6)
    assert output["safe_logits"].shape == (2, 6)
    assert output["arm_ordered_trust"].shape == (2, 126, 1)
    assert output["leg_ordered_trust"].shape == (2, 126, 1)
    torch.testing.assert_close(output["safe_logits"], swapped["safe_logits"], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(output["ordered_logits"], swapped["ordered_logits"])


def test_selective_dual_expert_losses_backpropagate() -> None:
    torch.manual_seed(37)
    model = MultiLimbHARModel(
        input_channels=9,
        num_classes=4,
        hidden_dim=12,
        encoder_blocks=1,
        dropout=0.0,
        temporal="tcn",
        variant="rcbd",
        role_vector=True,
        sample_rate=25,
        scales_seconds=[1, 2],
        tokens_per_scale=4,
        orientation_enabled=False,
        orientation_temperature=1.0,
        bidirectional_scales=False,
        anchor_mode="selective_dual",
    )
    limbs = [torch.randn(2, 40, 9) for _ in range(4)]
    anchor = torch.randn(2, 40, 9)
    output = model(*limbs, anchor=anchor)
    swapped = model(limbs[1], limbs[0], limbs[3], limbs[2], anchor=anchor)
    loss, parts = regularized_classification_loss(
        output,
        swapped,
        torch.tensor([0, 1]),
        None,
        {
            "swap_classification": 0.5,
            "selection_gate": 0.1,
            "ordered_classification": 0.5,
            "safe_classification": 0.5,
        },
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert {"swap_ce", "selection_gate", "ordered_ce", "safe_ce"}.issubset(parts)
    assert any(parameter.grad is not None for parameter in model.ordered_classifier.parameters())
    assert any(parameter.grad is not None for parameter in model.classifier.parameters())


def test_hard_route_gate_is_trained_only_by_explicit_gate_supervision() -> None:
    torch.manual_seed(41)
    model = MultiLimbHARModel(
        input_channels=9,
        num_classes=4,
        hidden_dim=12,
        encoder_blocks=1,
        dropout=0.0,
        temporal="tcn",
        variant="rcbd",
        role_vector=True,
        sample_rate=25,
        scales_seconds=[1, 2],
        tokens_per_scale=4,
        orientation_enabled=False,
        orientation_temperature=1.0,
        bidirectional_scales=False,
        anchor_mode="selective_dual",
        dual_encoder_sharing=False,
        dual_hard_routing=True,
    )
    limbs = [torch.randn(2, 40, 9) for _ in range(4)]
    anchor = torch.randn(2, 40, 9)
    output = model(*limbs, anchor=anchor)
    output["logits"].sum().backward()
    assert all(parameter.grad is None for parameter in model.arm_side_trust.parameters())

    model.zero_grad(set_to_none=True)
    output = model(*limbs, anchor=anchor)
    swapped = model(limbs[1], limbs[0], limbs[3], limbs[2], anchor=anchor)
    loss, _ = regularized_classification_loss(
        output,
        swapped,
        torch.tensor([0, 1]),
        None,
        {"selection_gate": 0.1},
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.arm_side_trust.parameters())


def test_validation_selection_score_supports_clean_and_robust_objectives() -> None:
    metrics = {"macro_f1": 0.9, "swap_stress": {"macro_f1": 0.6}}
    assert validation_selection_score(metrics) == 0.9
    assert validation_selection_score(metrics, "worst_case_macro_f1") == 0.6
    assert abs(validation_selection_score(metrics, "harmonic_macro_f1") - 0.72) < 1e-12


def test_audio_fusion_forward_shape() -> None:
    model = BilateralHARModel(
        input_channels=6,
        num_classes=5,
        hidden_dim=16,
        encoder_blocks=1,
        use_audio=True,
    ).eval()
    left = torch.randn(2, 30, 6)
    right = torch.randn(2, 30, 6)
    audio = torch.randn(2, 32, 40)
    with torch.no_grad():
        output = model(left, right, audio)
    assert output["logits"].shape == (2, 5)
    assert output["fusion_weights"].shape == (2, 2)
    torch.testing.assert_close(output["fusion_weights"].sum(dim=-1), torch.ones(2))


def test_orientation_candidates_are_24_proper_rotations() -> None:
    rotations = proper_signed_permutation_matrices()
    assert rotations.shape == (24, 3, 3)
    torch.testing.assert_close(torch.det(rotations), torch.ones(24))


def test_orientation_canonicalizer_shapes() -> None:
    output = LatentOrientationCanonicalizer(hidden_dim=16)(torch.randn(4, 30, 3))
    assert output.canonical.shape == (4, 30, 3)
    assert output.probabilities.shape == (4, 24)
    assert output.selected_index.shape == (4,)


def test_multilimb_rcbd_is_independently_pair_swap_invariant() -> None:
    torch.manual_seed(7)
    model = MultiLimbHARModel(
        input_channels=3,
        num_classes=5,
        hidden_dim=16,
        encoder_blocks=1,
        dropout=0.0,
        temporal="tcn",
        variant="rcbd",
        role_vector=True,
        sample_rate=10,
        scales_seconds=[1, 2, 4],
        tokens_per_scale=8,
        orientation_enabled=True,
        orientation_temperature=1.0,
        bidirectional_scales=True,
    ).eval()
    limbs = [torch.randn(2, 80, 3) for _ in range(4)]
    with torch.no_grad():
        reference = model(*limbs)["logits"]
        arm_swapped = model(limbs[1], limbs[0], limbs[2], limbs[3])["logits"]
        leg_swapped = model(limbs[0], limbs[1], limbs[3], limbs[2])["logits"]
    torch.testing.assert_close(reference, arm_swapped, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(reference, leg_swapped, atol=1e-6, rtol=1e-5)


def test_group_equivariant_orientation_canonicalizes_rotated_input() -> None:
    torch.manual_seed(17)
    model = LatentOrientationCanonicalizer(hidden_dim=16, mode="group_equivariant").eval()
    rotations = proper_signed_permutation_matrices()
    x = torch.randn(4, 60, 3)
    applied = torch.tensor([1, 5, 9, 13])
    rotated = torch.einsum("bij,btj->bti", rotations[applied], x)
    with torch.no_grad():
        original_output = model(x)
        rotated_output = model(rotated)
    targets = orientation_equivariance_targets(original_output.selected_index, applied, rotations)
    torch.testing.assert_close(rotated_output.selected_index, targets)
    torch.testing.assert_close(original_output.canonical, rotated_output.canonical)


def test_bidirectional_multiscale_uses_separate_past_and_future() -> None:
    aggregator = MultiScaleTemporalAggregator(
        input_dim=1,
        hidden_dim=4,
        dropout=0.0,
        kind="tcn",
        scales_seconds=[1],
        sample_rate=1,
        tokens_per_scale=1,
        bidirectional_windows=True,
    )
    signal = torch.arange(8, dtype=torch.float32).view(1, 8, 1)
    torch.testing.assert_close(
        aggregator._crop_and_pool(signal, 1, "past"), torch.tensor([[[3.0]]])
    )
    torch.testing.assert_close(
        aggregator._crop_and_pool(signal, 1, "future"), torch.tensor([[[4.0]]])
    )
    _, weights = aggregator(signal)
    assert weights.shape == (1, 2, 1)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 2))
def test_uniform_pair_rotation_supports_bilateral_batches() -> None:
    batch = {
        "left": torch.tensor([[[0.0, 0.0, 1.0]]] * 8),
        "right": torch.tensor([[[0.0, 0.0, 1.0]]] * 8),
    }
    torch.manual_seed(13)
    rotated, targets = rotate_one_side_per_pair(batch, probability=1.0, max_degrees=180.0)

    left_changed = (rotated["left"] != batch["left"]).any(dim=(1, 2))
    right_changed = (rotated["right"] != batch["right"]).any(dim=(1, 2))
    assert bool((left_changed ^ right_changed).all().item())
    assert set(targets) == {"arm"}
    assert "leg_quality_target" not in rotated
