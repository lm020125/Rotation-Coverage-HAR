from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_temporal_decoder_cv import evaluate_labels  # noqa: E402
from run_ul_catboost_cv import load_folds  # noqa: E402
from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.metrics import classification_metrics  # noqa: E402
from rcbd_har.multi_expert_selector import (  # noqa: E402
    apply_multi_expert_selector,
    build_multi_expert_features,
    multi_expert_targets,
)
from rcbd_har.utils import save_json  # noqa: E402


def score_key(archive) -> str:
    return "gate_score" if "gate_score" in archive.files else "gate_main_better_probability"


def train_selector(features: np.ndarray, targets: np.ndarray, weighting: str, args):
    from catboost import CatBoostClassifier

    if len(np.unique(targets)) != 3:
        raise ValueError("Multi-expert selector requires all three actions")
    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2,
        auto_class_weights="Balanced" if weighting == "balanced" else None,
        random_seed=args.seed,
        task_type="CPU",
        thread_count=args.thread_count,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(features, targets)
    return model


def cross_fitted_probabilities(
    features: np.ndarray,
    targets: np.ndarray,
    frame,
    folds: list[list[str]],
    weighting: str,
    args,
) -> np.ndarray:
    output = np.full((len(frame), 3), np.nan, dtype=np.float32)
    participants = frame.participant.astype(str).to_numpy()
    for validation_people in folds:
        validation = np.isin(participants, validation_people)
        model = train_selector(features[~validation], targets[~validation], weighting, args)
        output[validation] = model.predict_proba(features[validation]).astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("Cross-fitted selector probabilities are incomplete")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict raw/relational residual selector")
    parser.add_argument("--manifest", default="data/wear/manifest.csv")
    parser.add_argument("--raw-output", default="outputs/wear_da_ccfre_seed0")
    parser.add_argument(
        "--relational-output", default="outputs/wear_da_ccfre_relational_seed0"
    )
    parser.add_argument("--output", default="outputs/wear_multi_expert_selector_seed0")
    parser.add_argument("--folds", default="outputs/wear_protocol_decoder_dense_seed0/folds.json")
    parser.add_argument("--inner-root", default="outputs/wear_null_residual_meta_seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=10.0)
    parser.add_argument("--thread-count", type=int, default=8)
    parser.add_argument("--weightings", nargs="+", default=["balanced", "none"])
    parser.add_argument("--margins", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--minimum-inner-gain", type=float, default=1e-3)
    args = parser.parse_args()

    frame = load_manifest(args.manifest).reset_index(drop=True)
    targets = frame.label.to_numpy(np.int64)
    participant_values = frame.participant.astype(str).to_numpy()
    num_classes = int(targets.max()) + 1
    folds = load_folds(args.folds, participant_values.tolist(), 3)
    raw_root = Path(args.raw_output).resolve()
    relational_root = Path(args.relational_output).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_json({"test_folds": folds}, output / "folds.json")
    raw_outer = np.load(raw_root / "oof_predictions.npz")
    relational_outer = np.load(relational_root / "oof_predictions.npz")
    for archive in (raw_outer, relational_outer):
        if not np.array_equal(archive["row_indices"], np.arange(len(frame))):
            raise ValueError("Outer selector rows do not match the manifest")
        if not np.array_equal(archive["targets"], targets):
            raise ValueError("Outer selector labels do not match the manifest")

    selected_oof = np.full(len(frame), -1, dtype=np.int64)
    action_oof = np.full(len(frame), -1, dtype=np.int64)
    fold_summaries: list[dict[str, object]] = []
    for fold_index, test_people in enumerate(folds, start=1):
        raw_inner = np.load(raw_root / f"fold_{fold_index}" / "inner_routing_oof.npz")
        relational_inner = np.load(
            relational_root / f"fold_{fold_index}" / "inner_routing_oof.npz"
        )
        if not np.array_equal(raw_inner["row_indices"], relational_inner["row_indices"]):
            raise ValueError(f"Inner selector rows differ in fold {fold_index}")
        inner_rows = raw_inner["row_indices"].astype(np.int64)
        inner_frame = frame.iloc[inner_rows].reset_index(drop=True)
        inner_targets = raw_inner["targets"].astype(np.int64)
        inner_main = raw_inner["main_predictions"].astype(np.int64)
        inner_raw = raw_inner["gated_predictions"].astype(np.int64)
        inner_relational = relational_inner["gated_predictions"].astype(np.int64)
        features = build_multi_expert_features(
            inner_main,
            inner_raw,
            inner_relational,
            raw_inner[score_key(raw_inner)],
            relational_inner[score_key(relational_inner)],
            inner_frame,
            num_classes,
        )
        action_targets = multi_expert_targets(
            inner_main, inner_raw, inner_relational, inner_targets
        )
        with (
            Path(args.inner_root).resolve() / f"fold_{fold_index}" / "summary.json"
        ).open(encoding="utf-8") as handle:
            inner_folds = json.load(handle)["inner_folds"]
        candidates: list[dict[str, object]] = []
        for weighting in args.weightings:
            probabilities = cross_fitted_probabilities(
                features, action_targets, inner_frame, inner_folds, weighting, args
            )
            for margin in args.margins:
                prediction = apply_multi_expert_selector(
                    inner_main,
                    inner_raw,
                    inner_relational,
                    probabilities,
                    margin,
                )
                candidates.append(
                    {
                        "weighting": weighting,
                        "margin": margin,
                        "probabilities": probabilities,
                        "macro_f1": classification_metrics(
                            inner_targets, prediction, num_classes
                        )["macro_f1"],
                        "switch_fraction": float(np.mean(prediction != inner_main)),
                    }
                )
        best = sorted(
            candidates,
            key=lambda value: (
                -float(value["macro_f1"]),
                float(value["margin"]),
                value["weighting"] != "balanced",
            ),
        )[0]
        main_score = classification_metrics(inner_targets, inner_main, num_classes)[
            "macro_f1"
        ]
        enabled = float(best["macro_f1"]) >= main_score + args.minimum_inner_gain
        final_model = train_selector(features, action_targets, str(best["weighting"]), args)

        test_rows = np.flatnonzero(np.isin(participant_values, test_people))
        test_frame = frame.iloc[test_rows].reset_index(drop=True)
        outer_main = raw_outer["main_predictions"][test_rows].astype(np.int64)
        outer_raw = raw_outer["gated_predictions"][test_rows].astype(np.int64)
        outer_relational = relational_outer["gated_predictions"][test_rows].astype(np.int64)
        test_features = build_multi_expert_features(
            outer_main,
            outer_raw,
            outer_relational,
            raw_outer[score_key(raw_outer)][test_rows],
            relational_outer[score_key(relational_outer)][test_rows],
            test_frame,
            num_classes,
        )
        probabilities = final_model.predict_proba(test_features).astype(np.float32)
        selected = (
            apply_multi_expert_selector(
                outer_main,
                outer_raw,
                outer_relational,
                probabilities,
                float(best["margin"]),
            )
            if enabled
            else outer_main
        )
        selected_oof[test_rows] = selected
        action_oof[test_rows] = probabilities.argmax(1)
        result = evaluate_labels(selected, test_frame, num_classes)
        fold_summary = {
            "fold": fold_index,
            "test_participants": test_people,
            "action_target_counts": np.bincount(action_targets, minlength=3).tolist(),
            "selected_weighting": best["weighting"],
            "selected_margin": best["margin"],
            "enabled": enabled,
            "inner_main_center_macro_f1": main_score,
            "inner_selected_center_macro_f1": best["macro_f1"],
            "outer_switch_fraction": float(np.mean(selected != outer_main)),
            "test": result,
        }
        fold_dir = output / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        final_model.save_model(str(fold_dir / "selector.cbm"))
        save_json(fold_summary, fold_dir / "summary.json")
        fold_summaries.append(fold_summary)
        print(
            f"fold {fold_index}: enabled={enabled}; weighting={best['weighting']}; "
            f"margin={best['margin']}; sample={result['samplewise']['macro_f1']:.4%}",
            flush=True,
        )

    if np.any(selected_oof < 0):
        raise RuntimeError("Multi-expert selector OOF is incomplete")
    overall = evaluate_labels(selected_oof, frame, num_classes)
    summary = {
        "method": "Strict raw-relational residual expert selector",
        "strict_outer_test_label_free": True,
        "folds": fold_summaries,
        "overall": overall,
    }
    save_json(summary, output / "summary.json")
    np.savez_compressed(
        output / "oof_predictions.npz",
        row_indices=np.arange(len(frame)),
        selected_predictions=selected_oof,
        selected_actions=action_oof,
        targets=targets,
    )
    print(
        f"overall center={overall['center']['macro_f1']:.4%}; "
        f"sample={overall['samplewise']['macro_f1']:.4%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
