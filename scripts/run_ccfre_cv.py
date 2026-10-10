from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_protocol_gate_cv import (  # noqa: E402
    cross_fitted_gate_probabilities,
    inner_candidates,
)
from run_temporal_decoder_cv import evaluate_labels  # noqa: E402
from run_ul_catboost_cv import load_folds  # noqa: E402
from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.metrics import classification_metrics  # noqa: E402
from rcbd_har.protocol_gate import (  # noqa: E402
    apply_protocol_gate,
    build_protocol_gate_features,
    hard_vote_with_probability_tiebreak,
    protocol_gate_training_targets,
)
from rcbd_har.residual_expert import (  # noqa: E402
    aggregate_pair_segment_scores,
    apply_pair_expert_logit_offsets,
    apply_residual_expert_gate,
    apply_residual_utility_gate,
    build_residual_expert_features,
    fit_pair_expert_logit_offsets,
    residual_expert_training_targets,
    residual_route_scope,
    residual_utility_training_targets,
    select_residual_expert_threshold,
    select_residual_utility_rule,
)
from rcbd_har.utils import participant_sort_key, save_json  # noqa: E402


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def train_residual_gate(features: np.ndarray, targets: np.ndarray, args):
    from catboost import CatBoostClassifier

    expected_classes = 3 if args.utility_gate else 2
    if len(np.unique(targets)) != expected_classes:
        raise ValueError(
            f"Residual expert gate training requires {expected_classes} decision classes"
        )
    model = CatBoostClassifier(
        loss_function="MultiClass" if args.utility_gate else "Logloss",
        eval_metric="MultiClass" if args.utility_gate else "Logloss",
        iterations=args.residual_iterations,
        depth=args.residual_depth,
        learning_rate=args.residual_learning_rate,
        l2_leaf_reg=args.residual_l2,
        auto_class_weights=(
            "Balanced" if args.residual_weighting == "balanced" else None
        ),
        random_seed=args.seed,
        task_type=args.gate_task_type,
        devices=args.devices if args.gate_task_type == "GPU" else None,
        thread_count=args.thread_count,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(features, targets)
    return model


def cross_fitted_residual_probabilities(
    features: np.ndarray,
    decisive: np.ndarray,
    decision_targets: np.ndarray,
    frame,
    inner_folds: list[list[str]],
    args,
) -> np.ndarray:
    output_shape = (len(frame), 3) if args.utility_gate else (len(frame),)
    output = np.full(output_shape, np.nan, dtype=np.float32)
    participant_values = frame.participant.astype(str).to_numpy()
    full_targets = np.full(len(frame), -1, dtype=np.int64)
    full_targets[decisive] = decision_targets
    for validation_people in inner_folds:
        validation = np.isin(participant_values, validation_people)
        training = (~validation) & decisive
        gate = train_residual_gate(features[training], full_targets[training], args)
        probabilities = gate.predict_proba(features[validation]).astype(np.float32)
        output[validation] = probabilities if args.utility_gate else probabilities[:, 1]
    if not np.isfinite(output).all():
        raise RuntimeError("Cross-fitted residual gate probabilities are incomplete")
    return output


def cross_fitted_pair_probabilities(
    main_better_probability: np.ndarray,
    main_predictions: np.ndarray,
    expert_predictions: np.ndarray,
    targets: np.ndarray,
    frame,
    inner_folds: list[list[str]],
    shrinkage: float,
    strength: float,
    num_classes: int,
) -> np.ndarray:
    output = np.full(len(frame), np.nan, dtype=np.float32)
    participant_values = frame.participant.astype(str).to_numpy()
    for validation_people in inner_folds:
        validation = np.isin(participant_values, validation_people)
        training = ~validation
        offsets = fit_pair_expert_logit_offsets(
            main_predictions[training],
            expert_predictions[training],
            targets[training],
            num_classes,
            shrinkage,
        )
        output[validation] = apply_pair_expert_logit_offsets(
            main_better_probability[validation],
            main_predictions[validation],
            expert_predictions[validation],
            offsets,
            strength,
            num_classes,
        )
    if not np.isfinite(output).all():
        raise RuntimeError("Cross-fitted pair calibration is incomplete")
    return output


def reconstruct_seed_inner_main(
    seed: int,
    fold_index: int,
    frame,
    train_people: list[str],
    num_classes: int,
    args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inner_root = Path(args.inner_template.format(seed=seed)).resolve()
    cstg_root = Path(args.cstg_template.format(seed=seed)).resolve()
    inner = np.load(inner_root / f"fold_{fold_index}" / "inner_oof.npz")
    rows = inner["row_indices"].astype(np.int64)
    inner_frame = frame.iloc[rows].reset_index(drop=True)
    targets = inner["targets"].astype(np.int64)
    if not np.array_equal(targets, inner_frame.label.to_numpy(np.int64)):
        raise ValueError(f"Seed {seed} fold {fold_index} inner labels do not align")
    cstg_summary = load_json(cstg_root / f"fold_{fold_index}" / "summary.json")
    seed_args = copy.copy(args)
    seed_args.seed = seed
    aligned, base, protocol = inner_candidates(
        inner["probabilities"],
        inner_frame,
        targets,
        train_people,
        cstg_summary["prior_configuration"],
        cstg_summary["protocol_configuration"],
        num_classes,
        seed_args,
    )
    protocol_features, _ = build_protocol_gate_features(
        inner["probabilities"],
        aligned,
        inner["pair_probabilities"],
        base,
        protocol,
        inner_frame,
    )
    decisive, decision_targets = protocol_gate_training_targets(
        base, protocol, targets
    )
    residual_summary = load_json(inner_root / f"fold_{fold_index}" / "summary.json")
    gate_probability = cross_fitted_gate_probabilities(
        protocol_features,
        decisive,
        decision_targets,
        inner_frame,
        residual_summary["inner_folds"],
        seed_args,
    )
    gated = apply_protocol_gate(
        base,
        protocol,
        gate_probability,
        float(cstg_summary["selected_threshold"]),
    )
    return rows, targets, aligned.astype(np.float32), gated


def load_outer_main(
    frame,
    seeds: list[int],
    ensemble_root: Path,
    cstg_template: str,
    protocol_template: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ensemble = np.load(ensemble_root / "oof_predictions.npz")
    expected_rows = np.arange(len(frame))
    targets = frame.label.to_numpy(np.int64)
    if not np.array_equal(ensemble["row_indices"], expected_rows):
        raise ValueError("Main ensemble rows do not match the manifest")
    if not np.array_equal(ensemble["targets"], targets):
        raise ValueError("Main ensemble targets do not match the manifest")
    predictions: list[np.ndarray] = []
    supports: list[np.ndarray] = []
    for seed in seeds:
        cstg = np.load(
            Path(cstg_template.format(seed=seed)).resolve() / "oof_predictions.npz"
        )
        protocol = np.load(
            Path(protocol_template.format(seed=seed)).resolve()
            / "oof_predictions.npz"
        )
        predictions.append(cstg["gated_predictions"].astype(np.int64))
        supports.append(protocol["probabilities"].astype(np.float32))
    prediction_stack = np.stack(predictions)
    support_stack = np.stack(supports)
    reconstructed = hard_vote_with_probability_tiebreak(
        prediction_stack, support_stack
    )
    archived = ensemble["voted_predictions"].astype(np.int64)
    if not np.array_equal(reconstructed, archived):
        raise RuntimeError("The reconstructed four-seed main ensemble has changed")
    return archived, support_stack.mean(axis=0), targets, prediction_stack, support_stack


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict confusion-conditioned full-body residual expert"
    )
    parser.add_argument("--manifest", default="data/wear/manifest.csv")
    parser.add_argument(
        "--ensemble-output", default="outputs/wear_cstg_safe_ensemble4"
    )
    parser.add_argument(
        "--cstg-template", default="outputs/wear_cstg_safe_seed{seed}"
    )
    parser.add_argument(
        "--protocol-template", default="outputs/wear_protocol_decoder_dense_seed{seed}"
    )
    parser.add_argument(
        "--inner-template", default="outputs/wear_null_residual_meta_seed{seed}"
    )
    parser.add_argument(
        "--expert-output", default="outputs/wear_decompose_expert_4s_seed0"
    )
    parser.add_argument(
        "--expert-description",
        default="DecomposeWHAR-inspired four-limb 4-second raw expert",
    )
    parser.add_argument(
        "--routed-main-output",
        help=(
            "Optional previously nested routed model. Its outer and per-fold inner "
            "gated predictions replace the fixed ensemble labels while the fixed "
            "ensemble probabilities remain available as label-free support features."
        ),
    )
    parser.add_argument(
        "--expert-inner-template",
        default="outputs/wear_decompose_expert_inner_outer{fold}_seed0",
    )
    parser.add_argument("--auxiliary-output")
    parser.add_argument("--auxiliary-inner-template")
    parser.add_argument("--output", default="outputs/wear_ccfre_seed0")
    parser.add_argument("--folds")
    parser.add_argument("--fold-indices", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 42, 3407, 114514])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gate-iterations", type=int, default=300)
    parser.add_argument("--gate-depth", type=int, default=5)
    parser.add_argument("--gate-learning-rate", type=float, default=0.05)
    parser.add_argument("--gate-l2", type=float, default=10.0)
    parser.add_argument("--residual-iterations", type=int, default=300)
    parser.add_argument("--residual-depth", type=int, default=5)
    parser.add_argument("--residual-learning-rate", type=float, default=0.05)
    parser.add_argument("--residual-l2", type=float, default=10.0)
    parser.add_argument(
        "--residual-weightings",
        nargs="+",
        choices=["balanced", "none"],
        default=["balanced", "none"],
    )
    parser.add_argument(
        "--disagreement-aware",
        action="store_true",
        help="Expose fixed-ensemble vote and probability disagreement to the gate",
    )
    parser.add_argument(
        "--utility-gate",
        action="store_true",
        help="Learn neutral/loss/gain probabilities on main-expert disagreements",
    )
    parser.add_argument(
        "--utility-loss-costs", nargs="+", type=float, default=[0.5, 1.0, 2.0]
    )
    parser.add_argument(
        "--utility-thresholds",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.1, 0.2, 0.3],
    )
    parser.add_argument(
        "--pair-risk",
        action="store_true",
        help="Apply inner-OOF hierarchical directed-pair prior correction",
    )
    parser.add_argument(
        "--pair-shrinkages", nargs="+", type=float, default=[20.0, 50.0, 100.0, 200.0]
    )
    parser.add_argument(
        "--pair-strengths", nargs="+", type=float, default=[0.5, 1.0]
    )
    parser.add_argument("--gate-task-type", choices=["GPU", "CPU"], default="CPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--thread-count", type=int, default=8)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument("--minimum-inner-gain", type=float, default=1e-3)
    parser.add_argument(
        "--routing-scopes",
        nargs="+",
        choices=["all", "focus"],
        default=["all"],
        help="Inner-OOF candidates controlling where the expert may intervene",
    )
    parser.add_argument(
        "--routing-focus-classes", nargs="+", type=int, default=[17, 18]
    )
    parser.add_argument(
        "--segment-aware",
        action="store_true",
        help="Select label-free directed-pair segment score shrinkage on inner OOF",
    )
    parser.add_argument(
        "--segment-blends", nargs="+", type=float, default=[0.0, 0.5, 1.0]
    )
    parser.add_argument(
        "--segment-minimum-lengths", nargs="+", type=int, default=[3, 5, 9]
    )
    parser.add_argument("--laplace", type=float, default=1e-2)
    parser.add_argument("--smooth-radius", type=int, default=10)
    parser.add_argument("--smooth-sigma", type=float, default=6.0)
    args = parser.parse_args()

    if args.pair_risk and args.utility_gate:
        raise ValueError("Pair-risk calibration currently requires the binary gate")
    if args.utility_gate and args.routing_scopes != ["all"]:
        raise ValueError("Focus routing currently requires the binary residual gate")

    frame = load_manifest(args.manifest).reset_index(drop=True)
    participant_values = frame.participant.astype(str).to_numpy()
    participants = sorted(np.unique(participant_values), key=participant_sort_key)
    num_classes = int(frame.label.max()) + 1
    ensemble_root = Path(args.ensemble_output).resolve()
    folds = load_folds(
        args.folds
        or str(
            Path(args.protocol_template.format(seed=args.seeds[0])).resolve()
            / "folds.json"
        ),
        participant_values.tolist(),
        3,
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_json({"test_folds": folds}, output / "folds.json")
    selected_folds = (
        {value for value in args.fold_indices}
        if args.fold_indices
        else set(range(1, len(folds) + 1))
    )
    if not selected_folds or any(value < 1 or value > len(folds) for value in selected_folds):
        raise ValueError("Fold indices are outside the available folds")
    (
        main_predictions,
        main_probabilities,
        targets,
        outer_member_predictions,
        outer_member_probabilities,
    ) = load_outer_main(
        frame,
        args.seeds,
        ensemble_root,
        args.cstg_template,
        args.protocol_template,
    )
    routed_main_root = None
    if args.routed_main_output:
        routed_main_root = Path(args.routed_main_output).resolve()
        routed_main_archive = np.load(routed_main_root / "oof_predictions.npz")
        if not np.array_equal(routed_main_archive["row_indices"], np.arange(len(frame))):
            raise ValueError("Routed outer main rows do not match the manifest")
        if not np.array_equal(routed_main_archive["targets"], targets):
            raise ValueError("Routed outer main labels do not match the manifest")
        main_predictions = routed_main_archive["gated_predictions"].astype(np.int64)
    expert_archive = np.load(Path(args.expert_output).resolve() / "oof_predictions.npz")
    if not np.array_equal(expert_archive["row_indices"], np.arange(len(frame))):
        raise ValueError("Outer expert rows do not match the manifest")
    if not np.array_equal(expert_archive["targets"], targets):
        raise ValueError("Outer expert labels do not match the manifest")
    expert_probabilities = expert_archive["smoothed_probabilities"].astype(np.float32)
    expert_predictions = expert_probabilities.argmax(1)
    if bool(args.auxiliary_output) != bool(args.auxiliary_inner_template):
        raise ValueError("Outer and inner auxiliary expert paths must be supplied together")
    auxiliary_probabilities = None
    auxiliary_predictions = None
    if args.auxiliary_output:
        auxiliary_archive = np.load(
            Path(args.auxiliary_output).resolve() / "oof_predictions.npz"
        )
        if not np.array_equal(auxiliary_archive["row_indices"], np.arange(len(frame))):
            raise ValueError("Outer auxiliary expert rows do not match the manifest")
        if not np.array_equal(auxiliary_archive["targets"], targets):
            raise ValueError("Outer auxiliary expert labels do not match the manifest")
        auxiliary_probabilities = auxiliary_archive["smoothed_probabilities"].astype(
            np.float32
        )
        auxiliary_predictions = auxiliary_probabilities.argmax(1)

    gated_oof = np.full(len(frame), -1, dtype=np.int64)
    gate_probability_oof = np.full(len(frame), np.nan, dtype=np.float32)
    fold_summaries: list[dict[str, object]] = []
    for fold_index, test_people in enumerate(folds, start=1):
        if fold_index not in selected_folds:
            continue
        train_people = [value for value in participants if value not in test_people]
        seed_rows: list[np.ndarray] = []
        seed_targets: list[np.ndarray] = []
        seed_support: list[np.ndarray] = []
        seed_predictions: list[np.ndarray] = []
        for seed in args.seeds:
            rows, inner_targets, support, predictions = reconstruct_seed_inner_main(
                seed, fold_index, frame, train_people, num_classes, args
            )
            seed_rows.append(rows)
            seed_targets.append(inner_targets)
            seed_support.append(support)
            seed_predictions.append(predictions)
        if any(not np.array_equal(seed_rows[0], value) for value in seed_rows[1:]):
            raise ValueError(f"Inner rows differ across seeds in fold {fold_index}")
        if any(
            not np.array_equal(seed_targets[0], value) for value in seed_targets[1:]
        ):
            raise ValueError(f"Inner targets differ across seeds in fold {fold_index}")
        inner_rows = seed_rows[0]
        inner_targets = seed_targets[0]
        inner_frame = frame.iloc[inner_rows].reset_index(drop=True)
        inner_main_probability = np.stack(seed_support).mean(axis=0)
        inner_main = hard_vote_with_probability_tiebreak(
            np.stack(seed_predictions), np.stack(seed_support)
        )
        if routed_main_root is not None:
            routed_inner_archive = np.load(
                routed_main_root / f"fold_{fold_index}" / "inner_routing_oof.npz"
            )
            if not np.array_equal(routed_inner_archive["row_indices"], inner_rows):
                raise ValueError(
                    f"Routed inner main rows do not align in fold {fold_index}"
                )
            if not np.array_equal(routed_inner_archive["targets"], inner_targets):
                raise ValueError(
                    f"Routed inner main targets do not align in fold {fold_index}"
                )
            inner_main = routed_inner_archive["gated_predictions"].astype(np.int64)
        inner_expert_archive = np.load(
            Path(args.expert_inner_template.format(fold=fold_index)).resolve()
            / "oof_predictions.npz"
        )
        if not np.array_equal(inner_expert_archive["row_indices"], inner_rows):
            raise ValueError(f"Inner expert rows do not align in fold {fold_index}")
        if not np.array_equal(inner_expert_archive["targets"], inner_targets):
            raise ValueError(f"Inner expert targets do not align in fold {fold_index}")
        inner_expert_probability = inner_expert_archive[
            "smoothed_probabilities"
        ].astype(np.float32)
        inner_expert = inner_expert_probability.argmax(1)
        inner_auxiliary_probability = None
        inner_auxiliary = None
        if args.auxiliary_inner_template:
            inner_auxiliary_archive = np.load(
                Path(args.auxiliary_inner_template.format(fold=fold_index)).resolve()
                / "oof_predictions.npz"
            )
            if not np.array_equal(inner_auxiliary_archive["row_indices"], inner_rows):
                raise ValueError(
                    f"Inner auxiliary expert rows do not align in fold {fold_index}"
                )
            if not np.array_equal(inner_auxiliary_archive["targets"], inner_targets):
                raise ValueError(
                    f"Inner auxiliary expert targets do not align in fold {fold_index}"
                )
            inner_auxiliary_probability = inner_auxiliary_archive[
                "smoothed_probabilities"
            ].astype(np.float32)
            inner_auxiliary = inner_auxiliary_probability.argmax(1)
        inner_features, feature_names = build_residual_expert_features(
            inner_main_probability,
            inner_expert_probability,
            inner_main,
            inner_expert,
            inner_frame,
            member_probabilities=(
                np.stack(seed_support) if args.disagreement_aware else None
            ),
            member_predictions=(
                np.stack(seed_predictions) if args.disagreement_aware else None
            ),
            auxiliary_probabilities=inner_auxiliary_probability,
            auxiliary_predictions=inner_auxiliary,
        )
        if args.utility_gate:
            decisive, decision_targets = residual_utility_training_targets(
                inner_main, inner_expert, inner_targets
            )
        else:
            decisive, decision_targets = residual_expert_training_targets(
                inner_main, inner_expert, inner_targets
            )
        residual_summary = load_json(
            Path(args.inner_template.format(seed=args.seeds[0])).resolve()
            / f"fold_{fold_index}"
            / "summary.json"
        )
        full_decision_targets = np.full(len(inner_frame), -1, dtype=np.int64)
        full_decision_targets[decisive] = decision_targets
        weighting_candidates: list[dict[str, object]] = []
        for routing_scope in args.routing_scopes:
            scope_eligible = residual_route_scope(
                inner_main,
                inner_expert,
                scope=routing_scope,
                focus_classes=args.routing_focus_classes,
            )
            scope_decisive = decisive & scope_eligible
            scope_targets = full_decision_targets[scope_decisive]
            expected_classes = 3 if args.utility_gate else 2
            if len(np.unique(scope_targets)) != expected_classes:
                continue
            for weighting in args.residual_weightings:
                weighting_args = copy.copy(args)
                weighting_args.residual_weighting = weighting
                candidate_oof = cross_fitted_residual_probabilities(
                    inner_features,
                    scope_decisive,
                    scope_targets,
                    inner_frame,
                    residual_summary["inner_folds"],
                    weighting_args,
                )
                if not args.utility_gate:
                    candidate_oof[~scope_eligible] = 1.0
                if args.utility_gate:
                    candidate_loss_cost, candidate_threshold, candidate_scores = (
                        select_residual_utility_rule(
                            inner_main,
                            inner_expert,
                            candidate_oof,
                            inner_targets,
                            args.utility_loss_costs,
                            args.utility_thresholds,
                            num_classes,
                            minimum_gain=args.minimum_inner_gain,
                        )
                    )
                    candidate_predictions, _ = apply_residual_utility_gate(
                        inner_main,
                        inner_expert,
                        candidate_oof,
                        candidate_loss_cost,
                        candidate_threshold,
                    )
                    candidate_variants = [
                        (
                            candidate_oof,
                            candidate_loss_cost,
                            candidate_threshold,
                            candidate_scores,
                            candidate_predictions,
                            None,
                            0.0,
                            1.0,
                            1,
                        )
                    ]
                else:
                    candidate_loss_cost = None
                    pair_configurations: list[tuple[float | None, float]] = [
                        (None, 0.0)
                    ]
                    if args.pair_risk:
                        pair_configurations += [
                            (float(shrinkage), float(strength))
                            for shrinkage in args.pair_shrinkages
                            for strength in args.pair_strengths
                        ]
                    candidate_variants = []
                    for pair_shrinkage, pair_strength in pair_configurations:
                        calibrated_oof = (
                            candidate_oof
                            if pair_strength == 0.0
                            else cross_fitted_pair_probabilities(
                                candidate_oof,
                                inner_main,
                                inner_expert,
                                inner_targets,
                                inner_frame,
                                residual_summary["inner_folds"],
                                float(pair_shrinkage),
                                pair_strength,
                                num_classes,
                            )
                        )
                        segment_configurations = [(1.0, 1)]
                        if args.segment_aware:
                            segment_configurations += [
                                (float(blend), int(minimum_length))
                                for blend in args.segment_blends
                                for minimum_length in args.segment_minimum_lengths
                                if float(blend) != 1.0
                            ]
                        for segment_blend, segment_minimum_length in segment_configurations:
                            segment_oof = aggregate_pair_segment_scores(
                                inner_main,
                                inner_expert,
                                calibrated_oof,
                                inner_frame,
                                blend=segment_blend,
                                minimum_length=segment_minimum_length,
                            )
                            candidate_threshold, candidate_scores = (
                                select_residual_expert_threshold(
                                    inner_main,
                                    inner_expert,
                                    segment_oof,
                                    inner_targets,
                                    args.thresholds,
                                    num_classes,
                                    minimum_gain=args.minimum_inner_gain,
                                )
                            )
                            candidate_predictions = apply_residual_expert_gate(
                                inner_main,
                                inner_expert,
                                segment_oof,
                                candidate_threshold,
                            )
                            candidate_variants.append(
                                (
                                    segment_oof,
                                    candidate_loss_cost,
                                    candidate_threshold,
                                    candidate_scores,
                                    candidate_predictions,
                                    pair_shrinkage,
                                    pair_strength,
                                    segment_blend,
                                    segment_minimum_length,
                                )
                            )
                for (
                    variant_oof,
                    variant_loss_cost,
                    variant_threshold,
                    variant_scores,
                    variant_predictions,
                    pair_shrinkage,
                    pair_strength,
                    segment_blend,
                    segment_minimum_length,
                ) in candidate_variants:
                    weighting_candidates.append(
                        {
                            "routing_scope": routing_scope,
                            "weighting": weighting,
                            "probabilities": variant_oof,
                            "loss_cost": variant_loss_cost,
                            "threshold": variant_threshold,
                            "threshold_scores": variant_scores,
                            "pair_shrinkage": pair_shrinkage,
                            "pair_strength": pair_strength,
                            "segment_blend": segment_blend,
                            "segment_minimum_length": segment_minimum_length,
                            "macro_f1": classification_metrics(
                                inner_targets, variant_predictions, num_classes
                            )["macro_f1"],
                        }
                    )
        if not weighting_candidates:
            raise RuntimeError(f"No valid residual routing candidate in fold {fold_index}")
        selected_weighting = sorted(
            weighting_candidates,
            key=lambda value: (
                -float(value["macro_f1"]),
                float(value["pair_strength"]),
                -float(value["segment_blend"]),
                int(value["segment_minimum_length"]),
                value["routing_scope"] != "all",
                value["weighting"] != "balanced",
            ),
        )[0]
        selected_routing_scope = str(selected_weighting["routing_scope"])
        gate_inner_oof = selected_weighting["probabilities"]
        selected_threshold = float(selected_weighting["threshold"])
        selected_loss_cost = selected_weighting["loss_cost"]
        selected_pair_shrinkage = selected_weighting["pair_shrinkage"]
        selected_pair_strength = float(selected_weighting["pair_strength"])
        selected_segment_blend = float(selected_weighting["segment_blend"])
        selected_segment_minimum_length = int(
            selected_weighting["segment_minimum_length"]
        )
        threshold_scores = selected_weighting["threshold_scores"]
        final_gate_args = copy.copy(args)
        final_gate_args.residual_weighting = selected_weighting["weighting"]
        selected_scope_eligible = residual_route_scope(
            inner_main,
            inner_expert,
            scope=selected_routing_scope,
            focus_classes=args.routing_focus_classes,
        )
        selected_decisive = decisive & selected_scope_eligible
        gate = train_residual_gate(
            inner_features[selected_decisive],
            full_decision_targets[selected_decisive],
            final_gate_args,
        )

        test_rows = np.flatnonzero(np.isin(participant_values, test_people))
        test_frame = frame.iloc[test_rows].reset_index(drop=True)
        test_features, _ = build_residual_expert_features(
            main_probabilities[test_rows],
            expert_probabilities[test_rows],
            main_predictions[test_rows],
            expert_predictions[test_rows],
            test_frame,
            member_probabilities=(
                outer_member_probabilities[:, test_rows]
                if args.disagreement_aware
                else None
            ),
            member_predictions=(
                outer_member_predictions[:, test_rows]
                if args.disagreement_aware
                else None
            ),
            auxiliary_probabilities=(
                auxiliary_probabilities[test_rows]
                if auxiliary_probabilities is not None
                else None
            ),
            auxiliary_predictions=(
                auxiliary_predictions[test_rows]
                if auxiliary_predictions is not None
                else None
            ),
        )
        predicted_gate_probabilities = gate.predict_proba(test_features).astype(
            np.float32
        )
        if args.utility_gate:
            gated, gate_probability = apply_residual_utility_gate(
                main_predictions[test_rows],
                expert_predictions[test_rows],
                predicted_gate_probabilities,
                float(selected_loss_cost),
                selected_threshold,
            )
            inner_gated, inner_gate_score = apply_residual_utility_gate(
                inner_main,
                inner_expert,
                gate_inner_oof,
                float(selected_loss_cost),
                selected_threshold,
            )
        else:
            gate_probability = predicted_gate_probabilities[:, 1]
            test_scope_eligible = residual_route_scope(
                main_predictions[test_rows],
                expert_predictions[test_rows],
                scope=selected_routing_scope,
                focus_classes=args.routing_focus_classes,
            )
            gate_probability[~test_scope_eligible] = 1.0
            if selected_pair_strength > 0.0:
                pair_offsets = fit_pair_expert_logit_offsets(
                    inner_main,
                    inner_expert,
                    inner_targets,
                    num_classes,
                    float(selected_pair_shrinkage),
                )
                gate_probability = apply_pair_expert_logit_offsets(
                    gate_probability,
                    main_predictions[test_rows],
                    expert_predictions[test_rows],
                    pair_offsets,
                    selected_pair_strength,
                    num_classes,
                )
            gate_probability = aggregate_pair_segment_scores(
                main_predictions[test_rows],
                expert_predictions[test_rows],
                gate_probability,
                test_frame,
                blend=selected_segment_blend,
                minimum_length=selected_segment_minimum_length,
            )
            gated = apply_residual_expert_gate(
                main_predictions[test_rows],
                expert_predictions[test_rows],
                gate_probability,
                selected_threshold,
            )
            inner_gated = apply_residual_expert_gate(
                inner_main, inner_expert, gate_inner_oof, selected_threshold
            )
            inner_gate_score = gate_inner_oof
        gated_oof[test_rows] = gated
        gate_probability_oof[test_rows] = gate_probability
        test_result = evaluate_labels(gated, test_frame, num_classes)
        fold_summary = {
            "fold": fold_index,
            "train_participants": train_people,
            "test_participants": test_people,
            "feature_count": len(feature_names),
            "inner_training_count": int(decisive.sum()),
            "inner_selected_training_count": int(selected_decisive.sum()),
            "inner_decisive_count": (
                None if args.utility_gate else int(decisive.sum())
            ),
            "inner_main_better_fraction": (
                None if args.utility_gate else float(decision_targets.mean())
            ),
            "inner_utility_target_counts": (
                np.bincount(decision_targets, minlength=3).tolist()
                if args.utility_gate
                else None
            ),
            "selected_weighting": selected_weighting["weighting"],
            "selected_routing_scope": selected_routing_scope,
            "selected_loss_cost": selected_loss_cost,
            "selected_pair_shrinkage": selected_pair_shrinkage,
            "selected_pair_strength": selected_pair_strength,
            "selected_segment_blend": selected_segment_blend,
            "selected_segment_minimum_length": selected_segment_minimum_length,
            "weighting_selection_scores": [
                {
                    "weighting": value["weighting"],
                    "routing_scope": value["routing_scope"],
                    "inner_center_macro_f1": value["macro_f1"],
                    "selected_threshold": value["threshold"],
                    "selected_loss_cost": value["loss_cost"],
                    "pair_shrinkage": value["pair_shrinkage"],
                    "pair_strength": value["pair_strength"],
                    "segment_blend": value["segment_blend"],
                    "segment_minimum_length": value["segment_minimum_length"],
                }
                for value in weighting_candidates
            ],
            "selected_threshold": selected_threshold,
            "minimum_inner_gain": args.minimum_inner_gain,
            "threshold_selection_scores": threshold_scores,
            "inner_main_center_macro_f1": classification_metrics(
                inner_targets, inner_main, num_classes
            )["macro_f1"],
            "inner_expert_center_macro_f1": classification_metrics(
                inner_targets, inner_expert, num_classes
            )["macro_f1"],
            "inner_gated_center_macro_f1": classification_metrics(
                inner_targets, inner_gated, num_classes
            )["macro_f1"],
            "outer_expert_switch_fraction": float(
                np.mean(gated != main_predictions[test_rows])
            ),
            "test": test_result,
        }
        fold_dir = output / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        gate.save_model(str(fold_dir / "gate.cbm"))
        np.savez_compressed(
            fold_dir / "inner_routing_oof.npz",
            row_indices=inner_rows,
            main_predictions=inner_main,
            expert_predictions=inner_expert,
            gate_score=inner_gate_score,
            gated_predictions=inner_gated,
            targets=inner_targets,
        )
        save_json(fold_summary, fold_dir / "summary.json")
        fold_summaries.append(fold_summary)
        print(
            f"fold {fold_index}: threshold={selected_threshold:g}; "
            f"decisive={decisive.sum()}; "
            f"switch={fold_summary['outer_expert_switch_fraction']:.2%}; "
            f"center={test_result['center']['macro_f1']:.4%}; "
            f"sample={test_result['samplewise']['macro_f1']:.4%}",
            flush=True,
        )

    evaluated = gated_oof >= 0
    if not np.any(evaluated) or not np.isfinite(gate_probability_oof[evaluated]).all():
        raise RuntimeError("CCFRE OOF predictions are incomplete")
    evaluated_frame = frame.loc[evaluated].reset_index(drop=True)
    overall = evaluate_labels(gated_oof[evaluated], evaluated_frame, num_classes)
    summary = {
        "method": (
            "Pair-Calibrated Risk-Controlled Disagreement-Aware CCFRE "
            "(PCR-DA-CCFRE)"
            if args.disagreement_aware and args.pair_risk
            else "Disagreement-Aware Utility-Gated Full-Body Residual Expert "
            "(DAU-CCFRE)"
            if args.disagreement_aware and args.utility_gate
            else (
                "Segment-Aware Disagreement-Aware Confusion-Conditioned Full-Body "
                "Residual Expert (SA-DA-CCFRE)"
                if args.disagreement_aware and args.segment_aware
                else "Disagreement-Aware Confusion-Conditioned Full-Body Residual Expert "
                "(DA-CCFRE)"
                if args.disagreement_aware
                else "Confusion-Conditioned Full-Body Residual Expert (CCFRE)"
            )
        ),
        "strict_outer_test_label_free": True,
        "main_model": (
            f"nested routed predictions from {routed_main_root}"
            if routed_main_root is not None
            else "fixed equal hard vote of four CSTG seeds"
        ),
        "expert_model": args.expert_description,
        "auxiliary_expert": args.auxiliary_output,
        "gate_training": "participant-grouped outer-train inner OOF only",
        "threshold_selection": "cross-fitted outer-train participants only",
        "failure_fallback": "retain the unchanged main model unless inner gain exceeds the safety margin",
        "seeds": args.seeds,
        "evaluated_folds": sorted(selected_folds),
        "folds": fold_summaries,
        "overall": overall,
    }
    save_json(summary, output / "summary.json")
    np.savez_compressed(
        output / "oof_predictions.npz",
        row_indices=np.flatnonzero(evaluated),
        main_predictions=main_predictions[evaluated],
        expert_predictions=expert_predictions[evaluated],
        gate_score=gate_probability_oof[evaluated],
        gated_predictions=gated_oof[evaluated],
        targets=targets[evaluated],
    )
    print(
        f"overall center={overall['center']['macro_f1']:.4%}; "
        f"sample={overall['samplewise']['macro_f1']:.4%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
