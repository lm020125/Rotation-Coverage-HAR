from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Symmetric KL divergence used for left/right prediction consistency."""
    return symmetric_kl_per_sample(logits_a, logits_b).mean()


def symmetric_kl_per_sample(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Return one symmetric KL value per sample without reducing the batch."""
    if logits_a.shape != logits_b.shape or logits_a.ndim != 2:
        raise ValueError(
            "symmetric KL expects equal [batch, classes] logits, "
            f"got {logits_a.shape}/{logits_b.shape}"
        )
    log_a, log_b = F.log_softmax(logits_a, dim=-1), F.log_softmax(logits_b, dim=-1)
    prob_a, prob_b = log_a.exp(), log_b.exp()
    return 0.5 * (
        F.kl_div(log_a, prob_b, reduction="none").sum(dim=-1)
        + F.kl_div(log_b, prob_a, reduction="none").sum(dim=-1)
    )


def role_smoothness(role: torch.Tensor) -> torch.Tensor:
    if role.shape[1] < 2:
        return role.new_zeros(())
    return (role[:, 1:] - role[:, :-1]).abs().mean()


def role_utilization(role: torch.Tensor, minimum: float) -> torch.Tensor:
    """Discourage a completely collapsed scorer without forcing hard hand labels."""
    utilization = role.abs().mean()
    return torch.relu(role.new_tensor(minimum) - utilization)


def orientation_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1).mean()


def orientation_information_loss(probabilities: torch.Tensor) -> torch.Tensor:
    """Favor confident samples without forcing all samples into one orientation."""
    conditional = orientation_entropy(probabilities)
    marginal = probabilities.mean(dim=0)
    marginal_entropy = -(marginal * marginal.clamp_min(1e-8).log()).sum()
    return conditional - marginal_entropy


def regularized_classification_loss(
    output: dict[str, torch.Tensor],
    swapped_output: dict[str, torch.Tensor] | None,
    target: torch.Tensor,
    class_weights: torch.Tensor | None,
    loss_config: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    ce = F.cross_entropy(output["logits"], target, weight=class_weights)
    total = ce
    components = {"ce": float(ce.detach())}
    swap_weight = float(loss_config.get("swap_consistency", 0.0))
    if swap_weight and swapped_output is not None:
        swap = symmetric_kl(output["logits"], swapped_output["logits"])
        total = total + swap_weight * swap
        components["swap"] = float(swap.detach())
    swap_classification_weight = float(loss_config.get("swap_classification", 0.0))
    if swap_classification_weight:
        if swapped_output is None:
            raise ValueError("swap_classification requires a swapped model output")
        swapped_ce = F.cross_entropy(swapped_output["logits"], target, weight=class_weights)
        total = total + swap_classification_weight * swapped_ce
        components["swap_ce"] = float(swapped_ce.detach())
    gate_weight = float(loss_config.get("selection_gate", 0.0))
    if gate_weight:
        if swapped_output is None:
            raise ValueError("selection_gate requires a swapped model output")
        gate_keys = [key for key in output if key.endswith("_ordered_trust")]
        if not gate_keys or any(key not in swapped_output for key in gate_keys):
            raise ValueError("selection_gate requires matching ordered-trust diagnostics")
        clean_gate = torch.stack(
            [F.binary_cross_entropy(output[key], torch.ones_like(output[key])) for key in gate_keys]
        ).mean()
        swapped_gate = torch.stack(
            [
                F.binary_cross_entropy(swapped_output[key], torch.zeros_like(swapped_output[key]))
                for key in gate_keys
            ]
        ).mean()
        gate = 0.5 * (clean_gate + swapped_gate)
        total = total + gate_weight * gate
        components["selection_gate"] = float(gate.detach())
    ordered_weight = float(loss_config.get("ordered_classification", 0.0))
    if ordered_weight:
        if "ordered_logits" not in output:
            raise ValueError("ordered_classification requires ordered expert logits")
        ordered_ce = F.cross_entropy(output["ordered_logits"], target, weight=class_weights)
        total = total + ordered_weight * ordered_ce
        components["ordered_ce"] = float(ordered_ce.detach())
    safe_weight = float(loss_config.get("safe_classification", 0.0))
    if safe_weight:
        if "safe_logits" not in output:
            raise ValueError("safe_classification requires safe expert logits")
        safe_ce = F.cross_entropy(output["safe_logits"], target, weight=class_weights)
        total = total + safe_weight * safe_ce
        components["safe_ce"] = float(safe_ce.detach())
    role_keys = [key for key in output if key == "role_score" or key.endswith("_role_score")]
    if role_keys:
        smooth_weight = float(loss_config.get("role_smoothness", 0.0))
        if smooth_weight:
            smooth = torch.stack([role_smoothness(output[key]) for key in role_keys]).mean()
            total = total + smooth_weight * smooth
            components["role_smoothness"] = float(smooth.detach())
        utilization_weight = float(loss_config.get("role_utilization", 0.0))
        if utilization_weight:
            utilization = torch.stack(
                [
                    role_utilization(
                        output[key], float(loss_config.get("minimum_role_utilization", 0.05))
                    )
                    for key in role_keys
                ]
            ).mean()
            total = total + utilization_weight * utilization
            components["role_utilization"] = float(utilization.detach())
    orientation_keys = [key for key in output if key.endswith("orientation_probabilities")]
    orientation_weight = float(loss_config.get("orientation_entropy", 0.0))
    if orientation_keys and orientation_weight:
        entropy = torch.stack([orientation_entropy(output[key]) for key in orientation_keys]).mean()
        total = total + orientation_weight * entropy
        components["orientation_entropy"] = float(entropy.detach())
    components["total"] = float(total.detach())
    return total, components
