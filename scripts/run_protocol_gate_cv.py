from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_null_residual_cv import predict_ul_pairs  # noqa: E402
from run_prior_alignment_cv import align_participant  # noqa: E402
from run_protocol_decoder_cv import minimum_count_vector  # noqa: E402
from run_temporal_decoder_cv import evaluate_labels  # noqa: E402
from run_ul_catboost_cv import load_folds, load_participant  # noqa: E402
from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.engine import smooth_probabilities  # noqa: E402
from rcbd_har.metrics import classification_metrics  # noqa: E402
from rcbd_har.prior_alignment import mean_participant_prior  # noqa: E402
from rcbd_har.protocol_decoder import protocol_constrained_decode  # noqa: E402
from rcbd_har.protocol_gate import (  # noqa: E402
    apply_protocol_gate,
    build_protocol_gate_features,
    protocol_gate_training_targets,
    select_protocol_gate_threshold,
)
from rcbd_har.temporal_decoder import estimate_transition_matrix  # noqa: E402
from rcbd_har.utils import participant_sort_key, save_json  # noqa: E402


def train_protocol_gate(
    features: np.ndarray,
    targets: np.ndarray,
    args,
):
    from catboost import CatBoostClassifier

    if len(np.unique(targets)) != 2:
        raise ValueError("Protocol gate training requires both decision classes")
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=args.gate_iterations,
        depth=args.gate_depth,
        learning_rate=args.gate_learning_rate,
        l2_leaf_reg=args.gate_l2,
        auto_class_weights="Balanced",
        random_seed=args.seed,
        task_type=args.gate_task_type,
        devices=args.devices if args.gate_task_type == "GPU" else None,
        thread_count=args.thread_count,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(features, targets)
    return model


def inner_candidates(
    raw_probabilities: np.ndarray,
    frame,
    targets: np.ndarray,
    participants: list[str],
    prior_configuration: dict[str, object],
    protocol_configuration: dict[str, object],
    num_classes: int,
    args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    participant_values = frame.participant.astype(str).to_numpy()
    aligned = np.empty_like(raw_probabilities)
    decoded = np.full(len(frame), -1, dtype=np.int64)
    for participant in participants:
        selected = participant_values == participant
        source_people = [value for value in participants if value != participant]
        source = np.isin(participant_values, source_people)
        source_prior = mean_participant_prior(
            targets, participant_values, source_people, num_classes
        )
        participant_frame = frame.loc[selected].reset_index(drop=True)
        participant_aligned, _, _ = align_participant(
            raw_probabilities[selected],
            participant_frame,
            source_prior,
            str(prior_configuration["null_estimator"]),
            float(prior_configuration["rho"]),
            float(prior_configuration["beta"]),
            args,
        )
        aligned[selected] = participant_aligned
        transition = estimate_transition_matrix(
            targets[source],
            frame.loc[source].reset_index(drop=True),
            num_classes,
            args.laplace,
        )
        result = protocol_constrained_decode(
            participant_aligned,
            participant_frame,
            transition,
            float(protocol_configuration["strength"]),
            minimum_count_vector(
                num_classes, int(protocol_configuration["minimum_count"])
            ),
        )
        decoded[selected] = result.predictions
    if not np.isfinite(aligned).all() or np.any(decoded < 0):
        raise RuntimeError("Inner aligned or protocol predictions are incomplete")
    smoothed = smooth_probabilities(
        raw_probabilities, frame, args.smooth_radius, args.smooth_sigma
    )
    return aligned, smoothed.argmax(1), decoded


def cross_fitted_gate_probabilities(
    features: np.ndarray,
    decisive: np.ndarray,
    decision_targets: np.ndarray,
    frame,
    inner_folds: list[list[str]],
    args,
) -> np.ndarray:
    output = np.full(len(frame), np.nan, dtype=np.float32)
    participant_values = frame.participant.astype(str).to_numpy()
    decisive_targets = np.full(len(frame), -1, dtype=np.int64)
    decisive_targets[decisive] = decision_targets
    for validation_people in inner_folds:
        validation = np.isin(participant_values, validation_people)
        training = (~validation) & decisive
        model = train_protocol_gate(
            features[training], decisive_targets[training], args
        )
        output[validation] = model.predict_proba(features[validation])[:, 1]
    if not np.isfinite(output).all():
        raise RuntimeError("Cross-fitted protocol gate probabilities are incomplete")
    return output


def recompute_outer_pair_probabilities(
    base_root: Path,
    feature_root: Path,
    fold_index: int,
    test_people: list[str],
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(base_root / f"fold_{fold_index}" / "model.cbm"))
    data = [load_participant(feature_root, participant) for participant in test_people]
    rows = np.concatenate([value["row_indices"] for value in data]).astype(np.int64)
    features = np.concatenate([value["features"] for value in data])
    order = np.argsort(rows)
    return rows[order], predict_ul_pairs(model, features[order], num_classes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nested counterfactual sensor-temporal protocol trust gate"
    )
    parser.add_argument("--manifest", default="data/wear/manifest.csv")
    parser.add_argument("--features", default="data/wear/challenge_features")
    parser.add_argument("--base-output", default="outputs/wear_ul_catboost_seed0")
    parser.add_argument("--prior-output", default="outputs/wear_prior_alignment_seed0")
    parser.add_argument(
        "--protocol-output", default="outputs/wear_protocol_decoder_dense_seed0"
    )
    parser.add_argument("--inner-root", default="outputs/wear_null_residual_meta_seed0")
    parser.add_argument("--output", default="outputs/wear_cstg_seed0")
    parser.add_argument("--folds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gate-iterations", type=int, default=300)
    parser.add_argument("--gate-depth", type=int, default=5)
    parser.add_argument("--gate-learning-rate", type=float, default=0.05)
    parser.add_argument("--gate-l2", type=float, default=10.0)
    parser.add_argument("--gate-task-type", choices=["GPU", "CPU"], default="CPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--thread-count", type=int, default=8)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument("--minimum-inner-gain", type=float, default=1e-3)
    parser.add_argument("--laplace", type=float, default=1e-2)
    parser.add_argument("--smooth-radius", type=int, default=10)
    parser.add_argument("--smooth-sigma", type=float, default=6.0)
    args = parser.parse_args()

    frame = load_manifest(args.manifest).reset_index(drop=True)
    targets = frame.label.to_numpy(np.int64)
    participant_values = frame.participant.astype(str).to_numpy()
    participants = sorted(np.unique(participant_values), key=participant_sort_key)
    num_classes = int(targets.max()) + 1
    base_root = Path(args.base_output).resolve()
    prior_root = Path(args.prior_output).resolve()
    protocol_root = Path(args.protocol_output).resolve()
    inner_root = Path(args.inner_root).resolve()
    feature_root = Path(args.features).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    folds = load_folds(
        args.folds or str(protocol_root / "folds.json"),
        participant_values.tolist(),
        3,
    )
    save_json({"test_folds": folds}, output / "folds.json")

    base_archive = np.load(base_root / "oof_predictions.npz")
    prior_archive = np.load(prior_root / "oof_predictions.npz")
    protocol_archive = np.load(protocol_root / "oof_predictions.npz")
    for name, archive in (
        ("base", base_archive),
        ("prior", prior_archive),
        ("protocol", protocol_archive),
    ):
        if not np.array_equal(archive["row_indices"], np.arange(len(frame))):
            raise ValueError(f"{name} OOF rows do not match the manifest")
        if not np.array_equal(archive["targets"], targets):
            raise ValueError(f"{name} OOF targets do not match the manifest")
    raw_probabilities = base_archive["probabilities"].astype(np.float32)
    base_predictions = base_archive["smoothed_probabilities"].argmax(1)
    aligned_probabilities = prior_archive["aligned_probabilities"].astype(np.float32)
    protocol_predictions = protocol_archive["decoded_predictions"].astype(np.int64)
    if not np.allclose(
        protocol_archive["probabilities"], aligned_probabilities, atol=1e-7
    ):
        raise ValueError("Protocol emissions differ from prior-aligned probabilities")

    gated_oof = np.full(len(frame), -1, dtype=np.int64)
    gate_probability_oof = np.full(len(frame), np.nan, dtype=np.float32)
    fold_summaries = []
    for fold_index, test_people in enumerate(folds, start=1):
        train_people = [value for value in participants if value not in test_people]
        with (prior_root / f"fold_{fold_index}" / "summary.json").open(
            encoding="utf-8"
        ) as handle:
            prior_summary = json.load(handle)
        with (protocol_root / f"fold_{fold_index}" / "summary.json").open(
            encoding="utf-8"
        ) as handle:
            protocol_summary = json.load(handle)
        prior_configuration = prior_summary["selected_configuration"]
        protocol_configuration = protocol_summary["selected_configuration"]
        inner = np.load(inner_root / f"fold_{fold_index}" / "inner_oof.npz")
        inner_rows = inner["row_indices"].astype(np.int64)
        inner_frame = frame.iloc[inner_rows].reset_index(drop=True)
        inner_targets = inner["targets"].astype(np.int64)
        if not np.array_equal(inner_targets, inner_frame.label.to_numpy(np.int64)):
            raise ValueError(f"Inner targets do not align in fold {fold_index}")
        inner_aligned, inner_base, inner_protocol = inner_candidates(
            inner["probabilities"],
            inner_frame,
            inner_targets,
            train_people,
            prior_configuration,
            protocol_configuration,
            num_classes,
            args,
        )
        inner_features, feature_names = build_protocol_gate_features(
            inner["probabilities"],
            inner_aligned,
            inner["pair_probabilities"],
            inner_base,
            inner_protocol,
            inner_frame,
        )
        decisive, decision_targets = protocol_gate_training_targets(
            inner_base, inner_protocol, inner_targets
        )
        with (inner_root / f"fold_{fold_index}" / "summary.json").open(
            encoding="utf-8"
        ) as handle:
            residual_summary = json.load(handle)
        gate_inner_oof = cross_fitted_gate_probabilities(
            inner_features,
            decisive,
            decision_targets,
            inner_frame,
            residual_summary["inner_folds"],
            args,
        )
        selected_threshold, threshold_scores = select_protocol_gate_threshold(
            inner_base,
            inner_protocol,
            gate_inner_oof,
            inner_targets,
            args.thresholds,
            num_classes,
            minimum_gain=args.minimum_inner_gain,
        )
        gate = train_protocol_gate(
            inner_features[decisive], decision_targets, args
        )

        test_mask = np.isin(participant_values, test_people)
        test_rows = np.flatnonzero(test_mask)
        recomputed_rows, pair_probabilities = recompute_outer_pair_probabilities(
            base_root, feature_root, fold_index, test_people, num_classes
        )
        if not np.array_equal(recomputed_rows, test_rows):
            raise ValueError(f"Recomputed test rows differ in fold {fold_index}")
        pair_mean_difference = float(
            np.max(
                np.abs(pair_probabilities.mean(1) - raw_probabilities[test_rows])
            )
        )
        if pair_mean_difference > 2e-4:
            raise RuntimeError(
                f"Recomputed pair probabilities differ by {pair_mean_difference}"
            )
        test_frame = frame.iloc[test_rows].reset_index(drop=True)
        test_features, _ = build_protocol_gate_features(
            raw_probabilities[test_rows],
            aligned_probabilities[test_rows],
            pair_probabilities,
            base_predictions[test_rows],
            protocol_predictions[test_rows],
            test_frame,
        )
        gate_probability = gate.predict_proba(test_features)[:, 1].astype(np.float32)
        gated = apply_protocol_gate(
            base_predictions[test_rows],
            protocol_predictions[test_rows],
            gate_probability,
            selected_threshold,
        )
        gated_oof[test_rows] = gated
        gate_probability_oof[test_rows] = gate_probability
        fold_result = evaluate_labels(gated, test_frame, num_classes)
        inner_gated = apply_protocol_gate(
            inner_base, inner_protocol, gate_inner_oof, selected_threshold
        )
        fold_summary = {
            "fold": fold_index,
            "train_participants": train_people,
            "test_participants": test_people,
            "prior_configuration": prior_configuration,
            "protocol_configuration": protocol_configuration,
            "gate_feature_count": len(feature_names),
            "inner_decisive_count": int(decisive.sum()),
            "inner_base_better_fraction": float(decision_targets.mean()),
            "selected_threshold": selected_threshold,
            "minimum_inner_gain": args.minimum_inner_gain,
            "threshold_selection_scores": threshold_scores,
            "inner_base_center_macro_f1": classification_metrics(
                inner_targets, inner_base, num_classes
            )["macro_f1"],
            "inner_protocol_center_macro_f1": classification_metrics(
                inner_targets, inner_protocol, num_classes
            )["macro_f1"],
            "inner_gated_center_macro_f1": classification_metrics(
                inner_targets, inner_gated, num_classes
            )["macro_f1"],
            "outer_pair_recompute_max_abs_difference": pair_mean_difference,
            "outer_switch_fraction": float(
                np.mean(gated != protocol_predictions[test_rows])
            ),
            "test": fold_result,
        }
        fold_dir = output / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        gate.save_model(str(fold_dir / "gate.cbm"))
        save_json(fold_summary, fold_dir / "summary.json")
        fold_summaries.append(fold_summary)
        print(
            f"fold {fold_index}: threshold={selected_threshold:g}; "
            f"decisive={decisive.sum()}; switch={fold_summary['outer_switch_fraction']:.2%}; "
            f"center={fold_result['center']['macro_f1']:.4%}; "
            f"sample={fold_result['samplewise']['macro_f1']:.4%}",
            flush=True,
        )

    if np.any(gated_oof < 0) or not np.isfinite(gate_probability_oof).all():
        raise RuntimeError("CSTG OOF predictions are incomplete")
    overall = evaluate_labels(gated_oof, frame, num_classes)
    summary = {
        "method": "Counterfactual Sensor-Temporal Protocol Trust Gate",
        "strict_outer_test_label_free": True,
        "gate_training": "outer-train participant-grouped inner OOF only",
        "threshold_selection": "cross-fitted outer-train participants only",
        "counterfactual_views": "four upper-lower limb pair probabilities",
        "base_output": str(base_root),
        "prior_output": str(prior_root),
        "protocol_output": str(protocol_root),
        "folds": fold_summaries,
        "overall": overall,
    }
    save_json(summary, output / "summary.json")
    np.savez_compressed(
        output / "oof_predictions.npz",
        row_indices=np.arange(len(frame)),
        base_predictions=base_predictions,
        protocol_predictions=protocol_predictions,
        gate_base_better_probability=gate_probability_oof,
        gated_predictions=gated_oof,
        targets=targets,
    )
    print(
        f"overall center={overall['center']['macro_f1']:.4%}; "
        f"sample={overall['samplewise']['macro_f1']:.4%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
