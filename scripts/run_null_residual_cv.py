from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_ul_catboost_cv import (  # noqa: E402
    aligned_predict_proba,
    load_folds,
    load_participant,
    samplewise_targets,
)
from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.engine import expand_probabilities_to_samples, smooth_probabilities  # noqa: E402
from rcbd_har.metrics import classification_metrics  # noqa: E402
from rcbd_har.null_refinement import (  # noqa: E402
    build_null_gate_features,
    candidate_sample_weights,
    refine_null_probabilities,
)
from rcbd_har.ul_pairing import UL_PAIRS, make_grouped_folds, select_pair, stack_ul_training_variants  # noqa: E402
from rcbd_har.utils import participant_sort_key, save_json  # noqa: E402


def train_ul_model(train_data: list[dict[str, np.ndarray]], args):
    from catboost import CatBoostClassifier

    features = np.concatenate([value["features"] for value in train_data])
    labels = np.concatenate([value["labels"] for value in train_data])
    augmented, augmented_labels = stack_ul_training_variants(features, labels)
    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=args.base_iterations,
        depth=args.base_depth,
        border_count=args.border_count,
        auto_class_weights="Balanced",
        random_seed=args.seed,
        task_type=args.task_type,
        devices=args.devices if args.task_type == "GPU" else None,
        thread_count=args.thread_count,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(augmented, augmented_labels)
    return model


def predict_ul_pairs(model, features: np.ndarray, num_classes: int) -> np.ndarray:
    return np.stack(
        [
            aligned_predict_proba(model, select_pair(features, pair), num_classes)
            for pair in UL_PAIRS
        ],
        axis=1,
    )


def train_gate(
    features: np.ndarray,
    targets: np.ndarray,
    args,
    sample_weight: np.ndarray | None = None,
):
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=args.gate_iterations,
        depth=args.gate_depth,
        learning_rate=args.gate_learning_rate,
        l2_leaf_reg=args.gate_l2,
        random_seed=args.seed,
        task_type=args.task_type,
        devices=args.devices if args.task_type == "GPU" else None,
        thread_count=args.thread_count,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(features, targets, sample_weight=sample_weight)
    return model


def select_gate_configuration(
    gate_features: np.ndarray,
    candidates: np.ndarray,
    probabilities: np.ndarray,
    targets: np.ndarray,
    frame,
    inner_folds: list[list[str]],
    num_classes: int,
    args,
) -> tuple[float, str, list[dict[str, object]]]:
    configurations = [
        (alpha, weighting)
        for weighting in ("none", "candidate_balanced")
        for alpha in args.meta_alphas
    ]
    gate_predictions = {
        weighting: np.full(len(frame), np.nan, np.float32)
        for weighting in ("none", "candidate_balanced")
    }
    resolvable = (targets == 0) | (targets == candidates)
    for validation_people in inner_folds:
        validation = frame.participant.astype(str).isin(validation_people).to_numpy()
        meta_train = ~validation & resolvable
        active_targets = (targets[meta_train] != 0).astype(np.int64)
        for weighting in gate_predictions:
            weights = candidate_sample_weights(candidates, meta_train, weighting)
            model = train_gate(
                gate_features[meta_train], active_targets, args, sample_weight=weights
            )
            gate_predictions[weighting][validation] = model.predict_proba(
                gate_features[validation]
            )[:, 1]
    if any(not np.isfinite(values).all() for values in gate_predictions.values()):
        raise RuntimeError("Meta gate OOF predictions are incomplete")
    scores: list[dict[str, object]] = []
    for alpha, weighting in configurations:
        refined = refine_null_probabilities(
            probabilities, candidates, gate_predictions[weighting], alpha=alpha
        )
        smoothed = smooth_probabilities(
            refined, frame, args.smooth_radius, args.smooth_sigma
        )
        macro_f1 = classification_metrics(targets, smoothed.argmax(1), num_classes)[
            "macro_f1"
        ]
        scores.append(
            {"alpha": alpha, "weighting": weighting, "smoothed_center_macro_f1": macro_f1}
        )
    ranking = sorted(
        scores,
        key=lambda value: (
            -float(value["smoothed_center_macro_f1"]),
            value["weighting"] != "none",
            abs(float(value["alpha"]) - args.alpha),
        ),
    )
    best = ranking[0]
    return float(best["alpha"]), str(best["weighting"]), scores


def evaluate_probabilities(probabilities, frame, num_classes, args) -> tuple[dict, np.ndarray]:
    raw = classification_metrics(frame.label.to_numpy(), probabilities.argmax(1), num_classes)
    smoothed = smooth_probabilities(
        probabilities, frame, args.smooth_radius, args.smooth_sigma
    )
    smooth = classification_metrics(frame.label.to_numpy(), smoothed.argmax(1), num_classes)
    expanded = expand_probabilities_to_samples(smoothed, frame)
    sample_targets, sample_predictions = samplewise_targets(frame, expanded)
    sample = classification_metrics(sample_targets, sample_predictions, num_classes)
    return {"raw": raw, "smoothed": smooth, "samplewise": sample}, smoothed


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested OOF class-conditioned Null refinement")
    parser.add_argument("--manifest", default="data/wear/manifest.csv")
    parser.add_argument("--features", default="data/wear/challenge_features")
    parser.add_argument("--base-output", default="outputs/wear_ul_catboost_seed0")
    parser.add_argument("--output", default="outputs/wear_null_residual_seed0")
    parser.add_argument("--folds")
    parser.add_argument("--fold-indices", nargs="+", type=int)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-iterations", type=int, default=1000)
    parser.add_argument("--base-depth", type=int, default=5)
    parser.add_argument("--border-count", type=int, default=32)
    parser.add_argument("--gate-iterations", type=int, default=400)
    parser.add_argument("--gate-depth", type=int, default=5)
    parser.add_argument("--gate-learning-rate", type=float, default=0.05)
    parser.add_argument("--gate-l2", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--meta-select", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--meta-alphas", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--task-type", choices=["GPU", "CPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument("--smooth-radius", type=int, default=10)
    parser.add_argument("--smooth-sigma", type=float, default=6.0)
    args = parser.parse_args()

    from catboost import CatBoostClassifier

    frame = load_manifest(args.manifest).reset_index(drop=True)
    feature_dir = Path(args.features).resolve()
    base_output = Path(args.base_output).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    folds_path = args.folds or str(base_output / "folds.json")
    outer_folds = load_folds(folds_path, frame.participant.astype(str).tolist(), 3)
    selected = [value - 1 for value in args.fold_indices] if args.fold_indices else list(range(3))
    if not selected or any(value < 0 or value >= len(outer_folds) for value in selected):
        raise ValueError("fold-indices must be between 1 and 3")
    save_json({"test_folds": outer_folds}, output / "folds.json")
    participants = sorted(frame.participant.astype(str).unique(), key=participant_sort_key)
    num_classes = int(frame.label.max()) + 1
    outer_base_archive = np.load(base_output / "oof_predictions.npz")
    if not np.array_equal(outer_base_archive["row_indices"], np.arange(len(frame))):
        raise ValueError("Base OOF predictions do not cover the manifest in row order")
    saved_outer_probabilities = outer_base_archive["probabilities"]

    outputs = {
        "base": np.full((len(frame), num_classes), np.nan, np.float32),
        "refined": np.full((len(frame), num_classes), np.nan, np.float32),
        "gate_only": np.full((len(frame), num_classes), np.nan, np.float32),
    }
    gate_probabilities = np.full(len(frame), np.nan, np.float32)
    candidates_all = np.full(len(frame), -1, np.int64)
    fold_summaries = []

    for outer_index in selected:
        started = time.time()
        test_people = outer_folds[outer_index]
        train_people = [value for value in participants if value not in test_people]
        train_mask = frame.participant.astype(str).isin(train_people).to_numpy()
        train_rows = np.flatnonzero(train_mask)
        train_frame = frame.iloc[train_rows].reset_index(drop=True)
        inner_folds = make_grouped_folds(
            train_frame.participant.astype(str).tolist(), n_splits=args.inner_splits
        )
        inner_probabilities = np.full((len(train_rows), num_classes), np.nan, np.float32)
        inner_pairs = np.full((len(train_rows), len(UL_PAIRS), num_classes), np.nan, np.float32)
        train_row_lookup = {row: position for position, row in enumerate(train_rows)}
        print(
            f"outer {outer_index + 1}: train={train_people}, test={test_people}, "
            f"inner_folds={inner_folds}",
            flush=True,
        )
        for inner_index, validation_people in enumerate(inner_folds, start=1):
            inner_train_people = [value for value in train_people if value not in validation_people]
            model = train_ul_model(
                [load_participant(feature_dir, value) for value in inner_train_people], args
            )
            for participant in validation_people:
                data = load_participant(feature_dir, participant)
                pair_predictions = predict_ul_pairs(model, data["features"], num_classes)
                positions = np.asarray(
                    [train_row_lookup[int(row)] for row in data["row_indices"]], dtype=np.int64
                )
                inner_pairs[positions] = pair_predictions
                inner_probabilities[positions] = pair_predictions.mean(axis=1)
            print(
                f"outer {outer_index + 1} inner {inner_index}/{len(inner_folds)} complete",
                flush=True,
            )
        if not np.isfinite(inner_probabilities).all() or not np.isfinite(inner_pairs).all():
            raise RuntimeError("Inner OOF predictions are incomplete")
        gate_features, inner_candidates, feature_names = build_null_gate_features(
            inner_probabilities, inner_pairs, train_frame
        )
        train_targets = train_frame.label.to_numpy(np.int64)
        resolvable = (train_targets == 0) | (train_targets == inner_candidates)
        active_targets = (train_targets[resolvable] != 0).astype(np.int64)
        selected_alpha = args.alpha
        selected_weighting = "none"
        meta_scores: list[dict[str, object]] = []
        if args.meta_select:
            selected_alpha, selected_weighting, meta_scores = select_gate_configuration(
                gate_features,
                inner_candidates,
                inner_probabilities,
                train_targets,
                train_frame,
                inner_folds,
                num_classes,
                args,
            )
        final_weights = candidate_sample_weights(
            inner_candidates, resolvable, selected_weighting
        )
        gate = train_gate(
            gate_features[resolvable],
            active_targets,
            args,
            sample_weight=final_weights,
        )

        outer_model = CatBoostClassifier()
        outer_model.load_model(str(base_output / f"fold_{outer_index + 1}" / "model.cbm"))
        test_data = [load_participant(feature_dir, value) for value in test_people]
        test_rows = np.concatenate([value["row_indices"] for value in test_data]).astype(np.int64)
        order = np.argsort(test_rows)
        test_rows = test_rows[order]
        test_features = np.concatenate([value["features"] for value in test_data], axis=0)[order]
        test_pairs = predict_ul_pairs(outer_model, test_features, num_classes)
        test_probabilities = test_pairs.mean(axis=1)
        maximum_difference = float(
            np.max(np.abs(test_probabilities - saved_outer_probabilities[test_rows]))
        )
        if maximum_difference > 2e-4:
            raise RuntimeError(f"Recomputed outer probabilities differ by {maximum_difference}")
        test_frame = frame.iloc[test_rows].reset_index(drop=True)
        test_gate_features, test_candidates, _ = build_null_gate_features(
            test_probabilities, test_pairs, test_frame
        )
        active_probability = gate.predict_proba(test_gate_features)[:, 1].astype(np.float32)
        refined = refine_null_probabilities(
            test_probabilities, test_candidates, active_probability, alpha=selected_alpha
        )
        gate_only = refine_null_probabilities(
            test_probabilities, test_candidates, active_probability, alpha=1.0
        )
        outputs["base"][test_rows] = test_probabilities
        outputs["refined"][test_rows] = refined
        outputs["gate_only"][test_rows] = gate_only
        gate_probabilities[test_rows] = active_probability
        candidates_all[test_rows] = test_candidates
        fold_summary = {
            "fold": outer_index + 1,
            "train_participants": train_people,
            "test_participants": test_people,
            "inner_folds": inner_folds,
            "inner_resolvable_fraction": float(resolvable.mean()),
            "inner_active_fraction": float(active_targets.mean()),
            "base_recompute_max_abs_difference": maximum_difference,
            "gate_feature_count": len(feature_names),
            "selected_alpha": selected_alpha,
            "selected_weighting": selected_weighting,
            "meta_selection_scores": meta_scores,
            "elapsed_minutes": (time.time() - started) / 60,
        }
        fold_dir = output / f"fold_{outer_index + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        gate.save_model(str(fold_dir / "gate.cbm"))
        save_json(fold_summary, fold_dir / "summary.json")
        np.savez_compressed(
            fold_dir / "inner_oof.npz",
            row_indices=train_rows,
            probabilities=inner_probabilities,
            pair_probabilities=inner_pairs,
            candidates=inner_candidates,
            targets=train_targets,
        )
        fold_summaries.append(fold_summary)
        print(
            f"outer {outer_index + 1} gate complete: resolvable={resolvable.mean():.2%}, "
            f"active={active_targets.mean():.2%}, alpha={selected_alpha}, "
            f"weighting={selected_weighting}, elapsed={fold_summary['elapsed_minutes']:.1f} min",
            flush=True,
        )

    evaluated = np.isfinite(outputs["base"]).all(axis=1)
    evaluation_frame = frame.loc[evaluated].reset_index(drop=True)
    results = {}
    smoothed_outputs = {}
    for name, values in outputs.items():
        results[name], smoothed_outputs[name] = evaluate_probabilities(
            values[evaluated], evaluation_frame, num_classes, args
        )
    summary = {
        "method": "Nested class-conditioned Null residual refinement",
        "seed": args.seed,
        "alpha": args.alpha,
        "meta_select": args.meta_select,
        "meta_alphas": args.meta_alphas,
        "evaluated_folds": [value + 1 for value in selected],
        "strict_nested_oof": True,
        "folds": fold_summaries,
        "results": results,
    }
    save_json(summary, output / "summary.json")
    np.savez_compressed(
        output / "oof_predictions.npz",
        row_indices=np.flatnonzero(evaluated),
        base_probabilities=outputs["base"][evaluated],
        refined_probabilities=outputs["refined"][evaluated],
        gate_only_probabilities=outputs["gate_only"][evaluated],
        refined_smoothed_probabilities=smoothed_outputs["refined"],
        gate_active_probabilities=gate_probabilities[evaluated],
        candidates=candidates_all[evaluated],
        targets=evaluation_frame.label.to_numpy(np.int64),
    )
    print(
        f"base smooth={results['base']['smoothed']['macro_f1']:.4%}; "
        f"refined={results['refined']['smoothed']['macro_f1']:.4%}; "
        f"gate-only={results['gate_only']['smoothed']['macro_f1']:.4%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
