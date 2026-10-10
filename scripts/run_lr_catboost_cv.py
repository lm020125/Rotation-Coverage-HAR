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
from rcbd_har.ul_pairing import swap_left_right  # noqa: E402
from rcbd_har.utils import participant_sort_key, save_json  # noqa: E402


def evaluate(
    probabilities: np.ndarray,
    frame,
    num_classes: int,
    smooth_radius: int,
    smooth_sigma: float,
) -> tuple[dict, np.ndarray, dict]:
    raw = classification_metrics(frame.label.to_numpy(), probabilities.argmax(axis=1), num_classes)
    smoothed = smooth_probabilities(probabilities, frame, smooth_radius, smooth_sigma)
    smooth = classification_metrics(frame.label.to_numpy(), smoothed.argmax(axis=1), num_classes)
    expanded = expand_probabilities_to_samples(smoothed, frame)
    targets, predictions = samplewise_targets(frame, expanded)
    sample = classification_metrics(targets, predictions, num_classes)
    return {"raw": raw, "smoothed": smooth, "samplewise": sample}, smoothed, expanded


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-fold WEAR left-right CatBoost")
    parser.add_argument("--manifest", default="data/wear/manifest.csv")
    parser.add_argument("--features", default="data/wear/challenge_features")
    parser.add_argument("--output", default="outputs/wear_lr_catboost_seed0")
    parser.add_argument("--folds")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--fold-indices", nargs="+", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--border-count", type=int, default=32)
    parser.add_argument("--task-type", choices=["GPU", "CPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=50)
    parser.add_argument("--smooth-radius", type=int, default=10)
    parser.add_argument("--smooth-sigma", type=float, default=6.0)
    args = parser.parse_args()

    from catboost import CatBoostClassifier

    frame = load_manifest(args.manifest).reset_index(drop=True)
    feature_dir = Path(args.features).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    folds = load_folds(args.folds, frame.participant.astype(str).tolist(), args.n_splits)
    selected = [value - 1 for value in args.fold_indices] if args.fold_indices else list(range(len(folds)))
    if not selected or any(value < 0 or value >= len(folds) for value in selected):
        raise ValueError(f"Fold indices must be between 1 and {len(folds)}")
    save_json({"test_folds": folds}, output / "folds.json")

    participants = sorted(frame.participant.astype(str).unique(), key=participant_sort_key)
    num_classes = int(frame.label.max()) + 1
    oof_raw = np.full((len(frame), num_classes), np.nan, np.float32)
    oof_tta = np.full_like(oof_raw, np.nan)
    fold_summaries = []

    for fold_index in selected:
        started = time.time()
        test_people = folds[fold_index]
        train_people = [value for value in participants if value not in test_people]
        train_data = [load_participant(feature_dir, value) for value in train_people]
        original = np.concatenate([value["features"] for value in train_data])
        labels = np.concatenate([value["labels"] for value in train_data])
        augmented = np.concatenate([original, swap_left_right(original)])
        augmented_labels = np.tile(labels, 2)
        del original, labels, train_data
        model = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=args.iterations,
            depth=args.depth,
            border_count=args.border_count,
            auto_class_weights="Balanced",
            random_seed=args.seed,
            task_type=args.task_type,
            devices=args.devices if args.task_type == "GPU" else None,
            thread_count=args.thread_count,
            verbose=args.verbose,
            allow_writing_files=False,
        )
        print(
            f"fold {fold_index + 1}/{len(folds)}: train={train_people}, test={test_people}, "
            f"augmented={augmented.shape}",
            flush=True,
        )
        model.fit(augmented, augmented_labels)
        del augmented, augmented_labels
        fold_targets, fold_predictions = [], []
        for participant in test_people:
            data = load_participant(feature_dir, participant)
            raw = aligned_predict_proba(model, data["features"], num_classes)
            swapped = aligned_predict_proba(model, swap_left_right(data["features"]), num_classes)
            tta = (raw + swapped) / 2
            rows = data["row_indices"].astype(np.int64)
            oof_raw[rows] = raw
            oof_tta[rows] = tta
            fold_targets.append(data["labels"])
            fold_predictions.append(tta.argmax(axis=1))
        metrics = classification_metrics(
            np.concatenate(fold_targets), np.concatenate(fold_predictions), num_classes
        )
        fold_dir = output / f"fold_{fold_index + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(str(fold_dir / "model.cbm"))
        summary = {
            "fold": fold_index + 1,
            "train_participants": train_people,
            "test_participants": test_people,
            "tta_center_metrics_raw": metrics,
            "elapsed_minutes": (time.time() - started) / 60,
        }
        save_json(summary, fold_dir / "summary.json")
        fold_summaries.append(summary)
        print(f"fold {fold_index + 1} TTA raw macro-F1={metrics['macro_f1']:.4%}", flush=True)

    evaluated = np.isfinite(oof_raw).all(axis=1)
    evaluation_frame = frame.loc[evaluated].reset_index(drop=True)
    raw_probabilities = oof_raw[evaluated]
    tta_probabilities = oof_tta[evaluated]
    raw_result, raw_smoothed, _ = evaluate(
        raw_probabilities, evaluation_frame, num_classes, args.smooth_radius, args.smooth_sigma
    )
    tta_result, tta_smoothed, _ = evaluate(
        tta_probabilities, evaluation_frame, num_classes, args.smooth_radius, args.smooth_sigma
    )
    summary = {
        "method": "LR swap CatBoost",
        "seed": args.seed,
        "evaluated_folds": [value + 1 for value in selected],
        "folds": fold_summaries,
        "original": raw_result,
        "swap_tta": tta_result,
    }
    save_json(summary, output / "summary.json")
    np.savez_compressed(
        output / "oof_predictions.npz",
        row_indices=np.flatnonzero(evaluated),
        probabilities=raw_probabilities,
        tta_probabilities=tta_probabilities,
        smoothed_probabilities=raw_smoothed,
        tta_smoothed_probabilities=tta_smoothed,
        targets=evaluation_frame.label.to_numpy(np.int64),
    )
    print(
        f"LR original smooth={raw_result['smoothed']['macro_f1']:.4%}; "
        f"swap-TTA smooth={tta_result['smoothed']['macro_f1']:.4%}; "
        f"swap-TTA sample={tta_result['samplewise']['macro_f1']:.4%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
