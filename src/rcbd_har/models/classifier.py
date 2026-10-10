from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .encoders import (
    AudioCNNEncoder,
    MultiScaleTemporalAggregator,
    SharedIMUEncoder,
    TemporalAggregator,
)
from .deepconvlstm import DeepConvLSTM, FourLimbDeepConvLSTM
from .fusion import GatedMultimodalFusion
from .orientation import LatentOrientationCanonicalizer
from .rcbd import (
    AnchorReferencedRoleCanonicalBilateralDecomposition,
    RoleCanonicalBilateralDecomposition,
)


class BilateralHARModel(nn.Module):
    """Complete bilateral HAR model with RCBD and controlled baseline variants."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        hidden_dim: int = 128,
        encoder_blocks: int = 3,
        dropout: float = 0.2,
        temporal: str = "gru",
        variant: str = "rcbd",
        role_vector: bool = True,
        use_audio: bool = False,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.use_audio = use_audio
        self.wrist_encoder = SharedIMUEncoder(input_channels, hidden_dim, encoder_blocks, dropout)
        self.rcbd: RoleCanonicalBilateralDecomposition | None = None
        if variant == "rcbd":
            self.rcbd = RoleCanonicalBilateralDecomposition(hidden_dim, role_vector=role_vector)
            bilateral_dim = self.rcbd.output_dim
        elif variant in {"concat", "ordered_diff", "absdiff"}:
            bilateral_dim = hidden_dim * 2
        elif variant == "mean":
            bilateral_dim = hidden_dim
        elif variant == "deepsets":
            self.set_phi = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
            )
            bilateral_dim = hidden_dim
        else:
            raise ValueError(f"Unknown bilateral variant: {variant}")

        self.temporal_encoder = TemporalAggregator(bilateral_dim, hidden_dim, dropout, temporal)
        imu_embedding_dim = self.temporal_encoder.output_dim
        self.imu_projection = nn.Sequential(
            nn.Linear(imu_embedding_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        if use_audio:
            self.audio_encoder = AudioCNNEncoder(hidden_dim, dropout)
            self.fusion = GatedMultimodalFusion(hidden_dim, modalities=2, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        audio: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if left.shape != right.shape:
            raise ValueError(f"left/right shapes differ: {left.shape}/{right.shape}")
        batch = left.shape[0]
        encoded = self.wrist_encoder(torch.cat((left, right), dim=0))
        h_left, h_right = encoded[:batch], encoded[batch:]
        diagnostics: dict[str, torch.Tensor] = {}
        if self.variant == "rcbd":
            assert self.rcbd is not None
            output = self.rcbd(h_left, h_right)
            bilateral = output.representation
            diagnostics.update(
                role_score=output.role_score,
                common=output.common,
                raw_difference=output.raw_difference,
                canonical_difference=output.canonical_difference,
            )
        elif self.variant == "concat":
            bilateral = torch.cat((h_left, h_right), dim=-1)
        elif self.variant == "ordered_diff":
            common = 0.5 * (h_left + h_right)
            raw_difference = 0.5 * (h_left - h_right)
            bilateral = torch.cat((common, raw_difference), dim=-1)
            diagnostics.update(common=common, raw_difference=raw_difference)
        elif self.variant == "absdiff":
            common = 0.5 * (h_left + h_right)
            absolute_difference = 0.5 * (h_left - h_right).abs()
            bilateral = torch.cat((common, absolute_difference), dim=-1)
            diagnostics.update(common=common, absolute_difference=absolute_difference)
        elif self.variant == "mean":
            bilateral = 0.5 * (h_left + h_right)
        else:
            bilateral = 0.5 * (self.set_phi(h_left) + self.set_phi(h_right))

        imu_embedding = self.imu_projection(self.temporal_encoder(bilateral))
        final_embedding = imu_embedding
        if self.use_audio:
            if audio is None:
                raise ValueError("This model was configured with use_audio=true")
            audio_embedding = self.audio_encoder(audio)
            final_embedding, fusion_weights = self.fusion(imu_embedding, audio_embedding)
            diagnostics["fusion_weights"] = fusion_weights
        diagnostics["embedding"] = final_embedding
        diagnostics["logits"] = self.classifier(final_embedding)
        return diagnostics


class PairRepresentation(nn.Module):
    """Compose a left/right encoded pair using RCBD or a controlled baseline."""

    def __init__(
        self,
        hidden_dim: int,
        variant: str,
        role_vector: bool,
        anchor_referenced: bool = False,
        selective: bool = False,
        quality_mode: str = "learned",
        quality_gradient_mode: str = "joint",
    ) -> None:
        super().__init__()
        self.variant = variant
        self.anchor_referenced = anchor_referenced
        self.selective = selective
        self.quality_mode = str(quality_mode).lower()
        if self.quality_mode not in {"learned", "fixed_one", "fixed_zero"}:
            raise ValueError(f"Unknown bilateral quality mode: {quality_mode}")
        self.quality_gradient_mode = str(quality_gradient_mode).lower()
        if self.quality_gradient_mode not in {"joint", "isolated"}:
            raise ValueError(
                f"Unknown bilateral quality gradient mode: {quality_gradient_mode}"
            )
        if anchor_referenced and selective:
            raise ValueError("A pair cannot be both anchor-referenced and selective")
        self.rcbd: RoleCanonicalBilateralDecomposition | None = None
        self.anchor_rcbd: AnchorReferencedRoleCanonicalBilateralDecomposition | None = None
        self.quality_head: nn.Module | None = None
        if variant in {"rcbd", "qmbd"}:
            if variant == "qmbd" and (anchor_referenced or selective):
                raise ValueError("QMBD currently requires an unanchored, non-selective pair")
            if anchor_referenced:
                self.anchor_rcbd = AnchorReferencedRoleCanonicalBilateralDecomposition(
                    hidden_dim, role_vector
                )
            else:
                self.rcbd = RoleCanonicalBilateralDecomposition(hidden_dim, role_vector)
            if variant == "qmbd" and self.quality_mode == "learned":
                quality_hidden = max(16, hidden_dim // 2)
                self.quality_head = nn.Sequential(
                    nn.Linear(hidden_dim * 2, quality_hidden),
                    nn.GELU(),
                    nn.Linear(quality_hidden, 1),
                )
                nn.init.zeros_(self.quality_head[-1].weight)
                nn.init.constant_(self.quality_head[-1].bias, 2.9444389791664403)
            if selective:
                self.ordered_projection = nn.Linear(hidden_dim * 2, hidden_dim * 2)
                self.safe_projection = nn.Linear(hidden_dim * 2, hidden_dim * 2)
                self.side_trust = nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                nn.init.eye_(self.ordered_projection.weight)
                nn.init.zeros_(self.ordered_projection.bias)
                nn.init.eye_(self.safe_projection.weight)
                nn.init.zeros_(self.safe_projection.bias)
                nn.init.zeros_(self.side_trust[-1].weight)
                nn.init.constant_(self.side_trust[-1].bias, 2.0)
            self.output_dim = hidden_dim * 2
        elif anchor_referenced:
            raise ValueError("Anchor-referenced pair decomposition requires variant='rcbd'")
        elif variant in {"concat", "ordered_diff", "absdiff"}:
            self.output_dim = hidden_dim * 2
        elif variant == "mean":
            self.output_dim = hidden_dim
        elif variant == "deepsets":
            self.phi = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
            )
            self.output_dim = hidden_dim
        else:
            raise ValueError(f"Unknown pair variant: {variant}")

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        prefix: str,
        anchor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        diagnostics: dict[str, torch.Tensor] = {}
        if self.variant in {"rcbd", "qmbd"}:
            if self.anchor_referenced:
                if anchor is None:
                    raise ValueError("Anchor-referenced RCBD requires an encoded anchor")
                assert self.anchor_rcbd is not None
                output = self.anchor_rcbd(left, right, anchor)
            else:
                assert self.rcbd is not None
                output = self.rcbd(left, right)
            diagnostics.update(
                {
                    f"{prefix}_role_score": output.role_score,
                    f"{prefix}_common": output.common,
                    f"{prefix}_raw_difference": output.raw_difference,
                    f"{prefix}_canonical_difference": output.canonical_difference,
                }
            )
            if self.variant == "qmbd":
                if self.quality_mode == "learned":
                    assert self.quality_head is not None
                    quality_evidence = torch.cat(
                        (output.common, output.raw_difference.abs()), dim=-1
                    )
                    if self.quality_gradient_mode == "isolated":
                        quality_evidence = quality_evidence.detach()
                    quality = torch.sigmoid(self.quality_head(quality_evidence))
                elif self.quality_mode == "fixed_one":
                    quality = output.common.new_ones((*output.common.shape[:-1], 1))
                else:
                    quality = output.common.new_zeros((*output.common.shape[:-1], 1))
                diagnostics[f"{prefix}_quality"] = quality
                classification_quality = (
                    quality.detach()
                    if self.quality_gradient_mode == "isolated"
                    else quality
                )
                representation = torch.cat(
                    (output.common, classification_quality * output.canonical_difference),
                    dim=-1,
                )
                return representation, diagnostics
            if self.selective:
                if anchor is None:
                    raise ValueError("Selective RCBD requires an encoded anchor")
                ordered = self.ordered_projection(torch.cat((left, right), dim=-1))
                safe = self.safe_projection(output.representation)
                ordered_trust = torch.sigmoid(
                    self.side_trust(torch.cat((left, right, anchor), dim=-1))
                )
                diagnostics[f"{prefix}_ordered_trust"] = ordered_trust
                representation = ordered_trust * ordered + (1.0 - ordered_trust) * safe
                return representation, diagnostics
            return output.representation, diagnostics
        if self.variant == "concat":
            return torch.cat((left, right), dim=-1), diagnostics
        if self.variant == "ordered_diff":
            common = 0.5 * (left + right)
            raw_difference = 0.5 * (left - right)
            diagnostics.update(
                {
                    f"{prefix}_common": common,
                    f"{prefix}_raw_difference": raw_difference,
                }
            )
            return torch.cat((common, raw_difference), dim=-1), diagnostics
        if self.variant == "absdiff":
            common = 0.5 * (left + right)
            absolute_difference = 0.5 * (left - right).abs()
            diagnostics.update(
                {
                    f"{prefix}_common": common,
                    f"{prefix}_absolute_difference": absolute_difference,
                }
            )
            return torch.cat((common, absolute_difference), dim=-1), diagnostics
        if self.variant == "mean":
            return 0.5 * (left + right), diagnostics
        return 0.5 * (self.phi(left) + self.phi(right)), diagnostics


class MultiLimbHARModel(nn.Module):
    """WEAR-oriented arm/leg hierarchical RCBD with latent orientation canonicalization."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        hidden_dim: int,
        encoder_blocks: int,
        dropout: float,
        temporal: str,
        variant: str,
        role_vector: bool,
        sample_rate: float,
        scales_seconds: list[float],
        tokens_per_scale: int,
        orientation_enabled: bool,
        orientation_temperature: float,
        bidirectional_scales: bool = False,
        orientation_mode: str = "legacy",
        anchor_mode: str = "none",
        dual_encoder_sharing: bool = True,
        dual_hard_routing: bool = False,
        quality_mode: str = "learned",
        quality_gradient_mode: str = "joint",
    ) -> None:
        super().__init__()
        if orientation_enabled and input_channels != 3:
            raise ValueError("Latent orientation canonicalization currently requires 3-axis input")
        self.variant = variant
        self.orientation_enabled = orientation_enabled
        self.bidirectional_scales = bidirectional_scales
        self.anchor_mode = str(anchor_mode).lower()
        if self.anchor_mode not in {
            "none",
            "concat",
            "referenced",
            "selective",
            "selective_dual",
        }:
            raise ValueError(f"Unknown anchor_mode: {anchor_mode}")
        if self.anchor_mode == "referenced" and variant != "rcbd":
            raise ValueError("anchor_mode='referenced' requires variant='rcbd'")
        if self.anchor_mode == "selective" and variant != "rcbd":
            raise ValueError("anchor_mode='selective' requires variant='rcbd'")
        if self.anchor_mode == "selective_dual" and variant != "rcbd":
            raise ValueError("anchor_mode='selective_dual' requires variant='rcbd'")
        self.selective_dual = self.anchor_mode == "selective_dual"
        self.dual_encoder_sharing = bool(dual_encoder_sharing)
        self.dual_hard_routing = bool(dual_hard_routing)
        if orientation_enabled:
            self.arm_orientation = LatentOrientationCanonicalizer(
                hidden_dim=max(32, hidden_dim // 2),
                temperature=orientation_temperature,
                mode=orientation_mode,
            )
            self.leg_orientation = LatentOrientationCanonicalizer(
                hidden_dim=max(32, hidden_dim // 2),
                temperature=orientation_temperature,
                mode=orientation_mode,
            )
        self.arm_encoder = SharedIMUEncoder(input_channels, hidden_dim, encoder_blocks, dropout)
        self.leg_encoder = SharedIMUEncoder(input_channels, hidden_dim, encoder_blocks, dropout)
        anchor_referenced = self.anchor_mode == "referenced"
        selective = self.anchor_mode == "selective"
        self.arm_pair = PairRepresentation(
            hidden_dim,
            variant,
            role_vector,
            anchor_referenced=anchor_referenced,
            selective=selective,
            quality_mode=quality_mode,
            quality_gradient_mode=quality_gradient_mode,
        )
        self.leg_pair = PairRepresentation(
            hidden_dim,
            variant,
            role_vector,
            anchor_referenced=anchor_referenced,
            selective=selective,
            quality_mode=quality_mode,
            quality_gradient_mode=quality_gradient_mode,
        )
        if self.selective_dual:
            self.arm_ordered_pair = PairRepresentation(hidden_dim, "concat", role_vector)
            self.leg_ordered_pair = PairRepresentation(hidden_dim, "concat", role_vector)
            if not self.dual_encoder_sharing:
                self.arm_ordered_encoder = SharedIMUEncoder(
                    input_channels, hidden_dim, encoder_blocks, dropout
                )
                self.leg_ordered_encoder = SharedIMUEncoder(
                    input_channels, hidden_dim, encoder_blocks, dropout
                )
        self.arm_temporal = MultiScaleTemporalAggregator(
            self.arm_pair.output_dim,
            hidden_dim,
            dropout,
            temporal,
            scales_seconds,
            sample_rate,
            tokens_per_scale,
            bidirectional_windows=bidirectional_scales,
        )
        self.leg_temporal = MultiScaleTemporalAggregator(
            self.leg_pair.output_dim,
            hidden_dim,
            dropout,
            temporal,
            scales_seconds,
            sample_rate,
            tokens_per_scale,
            bidirectional_windows=bidirectional_scales,
        )
        if self.selective_dual:
            self.arm_ordered_temporal = MultiScaleTemporalAggregator(
                self.arm_ordered_pair.output_dim,
                hidden_dim,
                dropout,
                temporal,
                scales_seconds,
                sample_rate,
                tokens_per_scale,
                bidirectional_windows=bidirectional_scales,
            )
            self.leg_ordered_temporal = MultiScaleTemporalAggregator(
                self.leg_ordered_pair.output_dim,
                hidden_dim,
                dropout,
                temporal,
                scales_seconds,
                sample_rate,
                tokens_per_scale,
                bidirectional_windows=bidirectional_scales,
            )
        if self.anchor_mode != "none":
            self.anchor_encoder = SharedIMUEncoder(
                input_channels, hidden_dim, encoder_blocks, dropout
            )
            self.anchor_temporal = MultiScaleTemporalAggregator(
                hidden_dim,
                hidden_dim,
                dropout,
                temporal,
                scales_seconds,
                sample_rate,
                tokens_per_scale,
                bidirectional_windows=bidirectional_scales,
            )
            if self.selective_dual and not self.dual_encoder_sharing:
                self.anchor_ordered_encoder = SharedIMUEncoder(
                    input_channels, hidden_dim, encoder_blocks, dropout
                )
                self.anchor_ordered_temporal = MultiScaleTemporalAggregator(
                    hidden_dim,
                    hidden_dim,
                    dropout,
                    temporal,
                    scales_seconds,
                    sample_rate,
                    tokens_per_scale,
                    bidirectional_windows=bidirectional_scales,
                )
        pair_dim = self.arm_temporal.output_dim
        self.arm_projection = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.leg_projection = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        if self.anchor_mode != "none":
            self.anchor_projection = nn.Sequential(
                nn.Linear(self.anchor_temporal.output_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
        if self.selective_dual:
            self.arm_ordered_projection = nn.Sequential(
                nn.Linear(self.arm_ordered_temporal.output_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.leg_ordered_projection = nn.Sequential(
                nn.Linear(self.leg_ordered_temporal.output_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            if not self.dual_encoder_sharing:
                self.anchor_ordered_projection = nn.Sequential(
                    nn.Linear(self.anchor_ordered_temporal.output_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
        fusion_terms = 5 if self.anchor_mode != "none" else 4
        self.cross_pair_fusion = nn.Sequential(
            nn.Linear(hidden_dim * fusion_terms, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        if self.selective_dual:
            self.ordered_cross_pair_fusion = nn.Sequential(
                nn.Linear(hidden_dim * fusion_terms, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.arm_side_trust = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.leg_side_trust = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for gate in (self.arm_side_trust, self.leg_side_trust):
                nn.init.zeros_(gate[-1].weight)
                nn.init.constant_(gate[-1].bias, 2.0)
        missing_features = 5 if self.anchor_mode != "none" else 4
        self.missing_projection = nn.Linear(missing_features, hidden_dim, bias=False)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        if self.selective_dual:
            self.ordered_classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def _canonicalize_pair(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        canonicalizer: LatentOrientationCanonicalizer,
        prefix: str,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch = left.shape[0]
        output = canonicalizer(torch.cat((left, right), dim=0))
        diagnostics = {
            f"{prefix}_orientation_probabilities": output.probabilities,
            f"{prefix}_orientation_logits": output.logits,
            f"{prefix}_orientation_selected": output.selected_index,
            f"{prefix}_orientation_canonical": output.canonical,
        }
        return output.canonical[:batch], output.canonical[batch:], diagnostics

    def _encode_pair(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        encoder: SharedIMUEncoder,
        pair: PairRepresentation,
        prefix: str,
        anchor_sequence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, length = left.shape[:2]
        if not self.bidirectional_scales:
            encoded = encoder(torch.cat((left, right), dim=0))
            return pair(encoded[:batch], encoded[batch:], prefix, anchor_sequence)
        if length < 2:
            raise ValueError("Bidirectional encoding requires at least two time samples")
        if length % 2:
            # Official datasets such as DSADS contain odd-length windows (125
            # samples).  Repeat the final boundary sample so the past/future
            # branches remain equal without dropping any measured sample.
            left = torch.cat((left, left[:, -1:]), dim=1)
            right = torch.cat((right, right[:, -1:]), dim=1)
            length += 1
        center = length // 2
        encoded = encoder(
            torch.cat(
                (
                    left[:, :center],
                    right[:, :center],
                    left[:, center:],
                    right[:, center:],
                ),
                dim=0,
            )
        )
        left_past, right_past, left_future, right_future = encoded.split(batch, dim=0)
        anchor_past = anchor_future = None
        if anchor_sequence is not None:
            if anchor_sequence.shape[1] % 2:
                raise ValueError("Encoded anchor sequence must have an even time length")
            anchor_past, anchor_future = anchor_sequence.chunk(2, dim=1)
        past, past_diag = pair(left_past, right_past, prefix, anchor_past)
        future, future_diag = pair(left_future, right_future, prefix, anchor_future)
        diagnostics = {
            key: torch.cat((past_diag[key], future_diag[key]), dim=1) for key in past_diag
        }
        return torch.cat((past, future), dim=1), diagnostics

    def _encode_anchor(
        self,
        anchor: torch.Tensor,
        encoder: SharedIMUEncoder | None = None,
    ) -> torch.Tensor:
        encoder = self.anchor_encoder if encoder is None else encoder
        if not self.bidirectional_scales:
            return encoder(anchor)
        if anchor.shape[1] < 2:
            raise ValueError("Bidirectional anchor encoding requires at least two samples")
        if anchor.shape[1] % 2:
            anchor = torch.cat((anchor, anchor[:, -1:]), dim=1)
        center = anchor.shape[1] // 2
        encoded = encoder(torch.cat((anchor[:, :center], anchor[:, center:]), dim=0))
        past, future = encoded.chunk(2, dim=0)
        return torch.cat((past, future), dim=1)

    def forward(
        self,
        left_wrist: torch.Tensor,
        right_wrist: torch.Tensor,
        left_ankle: torch.Tensor,
        right_ankle: torch.Tensor,
        anchor: torch.Tensor | None = None,
        missing_fraction: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        shapes = {left_wrist.shape, right_wrist.shape, left_ankle.shape, right_ankle.shape}
        if len(shapes) != 1:
            raise ValueError("All four limb tensors must have the same [B,T,C] shape")
        if self.anchor_mode == "none" and anchor is not None:
            raise ValueError("This model was configured with anchor_mode='none'")
        if self.anchor_mode != "none":
            if anchor is None:
                raise ValueError(f"anchor_mode='{self.anchor_mode}' requires an anchor tensor")
            if anchor.shape != left_wrist.shape:
                raise ValueError(
                    f"Anchor must match limb [B,T,C] shape, got {anchor.shape}/{left_wrist.shape}"
                )
        diagnostics: dict[str, torch.Tensor] = {}
        if self.orientation_enabled:
            left_wrist, right_wrist, arm_diag = self._canonicalize_pair(
                left_wrist, right_wrist, self.arm_orientation, "arm"
            )
            left_ankle, right_ankle, leg_diag = self._canonicalize_pair(
                left_ankle, right_ankle, self.leg_orientation, "leg"
            )
            diagnostics.update(arm_diag)
            diagnostics.update(leg_diag)
        anchor_sequence = self._encode_anchor(anchor) if anchor is not None else None
        ordered_anchor_sequence = anchor_sequence
        if self.selective_dual and not self.dual_encoder_sharing:
            assert anchor is not None
            ordered_anchor_sequence = self._encode_anchor(anchor, self.anchor_ordered_encoder)
        arm_sequence, arm_diag = self._encode_pair(
            left_wrist,
            right_wrist,
            self.arm_encoder,
            self.arm_pair,
            "arm",
            anchor_sequence,
        )
        leg_sequence, leg_diag = self._encode_pair(
            left_ankle,
            right_ankle,
            self.leg_encoder,
            self.leg_pair,
            "leg",
            anchor_sequence,
        )
        arm_ordered_sequence = leg_ordered_sequence = None
        if self.selective_dual:
            ordered_arm_encoder = (
                self.arm_encoder
                if self.dual_encoder_sharing
                else self.arm_ordered_encoder
            )
            ordered_leg_encoder = (
                self.leg_encoder
                if self.dual_encoder_sharing
                else self.leg_ordered_encoder
            )
            arm_ordered_sequence, _ = self._encode_pair(
                left_wrist,
                right_wrist,
                ordered_arm_encoder,
                self.arm_ordered_pair,
                "arm_ordered",
                ordered_anchor_sequence,
            )
            leg_ordered_sequence, _ = self._encode_pair(
                left_ankle,
                right_ankle,
                ordered_leg_encoder,
                self.leg_ordered_pair,
                "leg_ordered",
                ordered_anchor_sequence,
            )
        diagnostics.update(arm_diag)
        diagnostics.update(leg_diag)
        arm_embedding, arm_scales = self.arm_temporal(arm_sequence)
        leg_embedding, leg_scales = self.leg_temporal(leg_sequence)
        arm_embedding = self.arm_projection(arm_embedding)
        leg_embedding = self.leg_projection(leg_embedding)
        arm_ordered_embedding = leg_ordered_embedding = None
        arm_ordered_scales = leg_ordered_scales = None
        if self.selective_dual:
            assert arm_ordered_sequence is not None and leg_ordered_sequence is not None
            arm_ordered_embedding, arm_ordered_scales = self.arm_ordered_temporal(
                arm_ordered_sequence
            )
            leg_ordered_embedding, leg_ordered_scales = self.leg_ordered_temporal(
                leg_ordered_sequence
            )
            arm_ordered_embedding = self.arm_ordered_projection(arm_ordered_embedding)
            leg_ordered_embedding = self.leg_ordered_projection(leg_ordered_embedding)
        fusion_inputs = [
            arm_embedding,
            leg_embedding,
            arm_embedding * leg_embedding,
            (arm_embedding - leg_embedding).abs(),
        ]
        anchor_embedding = None
        anchor_scales = None
        ordered_anchor_embedding = None
        if anchor_sequence is not None:
            anchor_embedding, anchor_scales = self.anchor_temporal(anchor_sequence)
            anchor_embedding = self.anchor_projection(anchor_embedding)
            fusion_inputs.insert(0, anchor_embedding)
            ordered_anchor_embedding = anchor_embedding
            if self.selective_dual and not self.dual_encoder_sharing:
                assert ordered_anchor_sequence is not None
                ordered_anchor_embedding, _ = self.anchor_ordered_temporal(
                    ordered_anchor_sequence
                )
                ordered_anchor_embedding = self.anchor_ordered_projection(
                    ordered_anchor_embedding
                )
        fused = self.cross_pair_fusion(torch.cat(fusion_inputs, dim=-1))
        ordered_fused = None
        if self.selective_dual:
            assert arm_ordered_embedding is not None and leg_ordered_embedding is not None
            ordered_fusion_inputs = [
                arm_ordered_embedding,
                leg_ordered_embedding,
                arm_ordered_embedding * leg_ordered_embedding,
                (arm_ordered_embedding - leg_ordered_embedding).abs(),
            ]
            if ordered_anchor_embedding is not None:
                ordered_fusion_inputs.insert(0, ordered_anchor_embedding)
            ordered_fused = self.ordered_cross_pair_fusion(
                torch.cat(ordered_fusion_inputs, dim=-1)
            )
        if missing_fraction is not None:
            expected_missing = 5 if self.anchor_mode != "none" else 4
            if missing_fraction.ndim != 2 or missing_fraction.shape[1] != expected_missing:
                raise ValueError(f"missing_fraction must have [B,{expected_missing}] shape")
            missing_terms = [
                missing_fraction[:, 0] + missing_fraction[:, 1],
                (missing_fraction[:, 0] - missing_fraction[:, 1]).abs(),
                missing_fraction[:, 2] + missing_fraction[:, 3],
                (missing_fraction[:, 2] - missing_fraction[:, 3]).abs(),
            ]
            if self.anchor_mode != "none":
                missing_terms.append(missing_fraction[:, 4])
            invariant_missing = torch.stack(missing_terms, dim=-1)
            fused = fused + self.missing_projection(invariant_missing)
            if ordered_fused is not None:
                ordered_fused = ordered_fused + self.missing_projection(invariant_missing)
            diagnostics["missing_features"] = invariant_missing
        safe_logits = self.classifier(fused)
        logits = safe_logits
        if self.selective_dual:
            assert ordered_fused is not None
            assert arm_ordered_sequence is not None and leg_ordered_sequence is not None
            assert ordered_anchor_sequence is not None
            arm_left, arm_right = arm_ordered_sequence.chunk(2, dim=-1)
            leg_left, leg_right = leg_ordered_sequence.chunk(2, dim=-1)
            arm_ordered_trust = torch.sigmoid(
                self.arm_side_trust(
                    torch.cat((arm_left, arm_right, ordered_anchor_sequence), dim=-1)
                )
            )
            leg_ordered_trust = torch.sigmoid(
                self.leg_side_trust(
                    torch.cat((leg_left, leg_right, ordered_anchor_sequence), dim=-1)
                )
            )
            ordered_trust = 0.5 * (
                arm_ordered_trust.mean(dim=1) + leg_ordered_trust.mean(dim=1)
            )
            ordered_logits = self.ordered_classifier(ordered_fused)
            ordered_route = ordered_trust
            if self.dual_hard_routing:
                # Routing is supervised explicitly by clean/swap gate targets.
                # Detaching the discrete decision prevents classification loss
                # from collapsing the risk detector onto whichever expert is
                # momentarily easier early in training.
                ordered_route = (ordered_trust >= 0.5).to(ordered_trust.dtype).detach()
            logits = ordered_route * ordered_logits + (1.0 - ordered_route) * safe_logits
            diagnostics.update(
                arm_ordered_trust=arm_ordered_trust,
                leg_ordered_trust=leg_ordered_trust,
                ordered_route=ordered_route,
                ordered_logits=ordered_logits,
                safe_logits=safe_logits,
                ordered_embedding=ordered_fused,
                arm_ordered_scale_weights=arm_ordered_scales,
                leg_ordered_scale_weights=leg_ordered_scales,
            )
        diagnostics.update(
            arm_scale_weights=arm_scales,
            leg_scale_weights=leg_scales,
            arm_embedding=arm_embedding,
            leg_embedding=leg_embedding,
            embedding=fused,
            logits=logits,
        )
        if anchor_embedding is not None:
            diagnostics["anchor_embedding"] = anchor_embedding
            assert anchor_scales is not None
            diagnostics["anchor_scale_weights"] = anchor_scales
        return diagnostics


def build_model(config: dict[str, Any]) -> nn.Module:
    data, model = config["data"], config["model"]
    if str(model.get("architecture", "rcbd_har")).lower() == "deepconvlstm":
        limb_mode = str(data.get("limb_mode", "bilateral"))
        model_class: type[nn.Module]
        if limb_mode == "bilateral":
            model_class = DeepConvLSTM
        elif limb_mode == "four":
            model_class = FourLimbDeepConvLSTM
        else:
            raise ValueError(f"Unsupported DeepConvLSTM limb_mode: {limb_mode}")
        return model_class(
            input_channels_per_sensor=int(data["imu_channels"]),
            num_classes=int(data["num_classes"]),
            conv_channels=int(model.get("conv_channels", 64)),
            lstm_hidden=int(model.get("lstm_hidden", 128)),
            lstm_layers=int(model.get("lstm_layers", 2)),
            dropout=float(model.get("dropout", 0.5)),
        )
    if str(data.get("limb_mode", "bilateral")) == "four":
        return MultiLimbHARModel(
            input_channels=int(data["imu_channels"]),
            num_classes=int(data["num_classes"]),
            hidden_dim=int(model.get("hidden_dim", 128)),
            encoder_blocks=int(model.get("encoder_blocks", 3)),
            dropout=float(model.get("dropout", 0.2)),
            temporal=str(model.get("temporal", "tcn")),
            variant=str(model.get("variant", "rcbd")),
            role_vector=bool(model.get("role_vector", True)),
            sample_rate=float(data.get("sample_rate", 50.0)),
            scales_seconds=[float(value) for value in model.get("scales_seconds", [1, 2, 4])],
            tokens_per_scale=int(model.get("tokens_per_scale", 64)),
            orientation_enabled=bool(model.get("orientation_enabled", True)),
            orientation_temperature=float(model.get("orientation_temperature", 1.0)),
            bidirectional_scales=bool(model.get("bidirectional_scales", False)),
            orientation_mode=str(model.get("orientation_mode", "legacy")),
            anchor_mode=str(model.get("anchor_mode", "none")),
            dual_encoder_sharing=bool(model.get("dual_encoder_sharing", True)),
            dual_hard_routing=bool(model.get("dual_hard_routing", False)),
            quality_mode=str(model.get("quality_mode", "learned")),
            quality_gradient_mode=str(model.get("quality_gradient_mode", "joint")),
        )
    return BilateralHARModel(
        input_channels=int(data["imu_channels"]),
        num_classes=int(data["num_classes"]),
        hidden_dim=int(model.get("hidden_dim", 128)),
        encoder_blocks=int(model.get("encoder_blocks", 3)),
        dropout=float(model.get("dropout", 0.2)),
        temporal=str(model.get("temporal", "gru")),
        variant=str(model.get("variant", "rcbd")),
        role_vector=bool(model.get("role_vector", True)),
        use_audio=bool(data.get("use_audio", False)),
    )
