from __future__ import annotations

import argparse
import json
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
from rcbd_har.challenge_features import LIMB_ORDER  # noqa: E402
from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.engine import expand_probabilities_to_samples, smooth_probabilities  # noqa: E402
from rcbd_har.metrics import classification_metrics  # noqa: E402
from rcbd_har.ul_pairing import swap_left_right  # noqa: E402
from rcbd_har.utils import participant_sort_key, save_json  # noqa: E402


def sensor_subset_indices(feature_count: int, limbs: list[str]) -> np.ndarray:
    if feature_count % len(LIMB_ORDER):
        raise ValueError("The challenge feature count must be divisible by four limbs")
    unknown = sorted(set(limbs).difference(LIMB_ORDER))
    if unknown:
        raise ValueError(f"Unknown limbs: {unknown}")
    if not limbs or len(set(limbs)) != len(limbs):
        raise ValueError("--limbs must contain distinct limb names")
    per_limb = feature_count // len(LIMB_ORDER)
    return np.concatenate(
        [
            np.arange(
                LIMB_ORDER.index(limb) * per_limb,
                (LIMB_ORDER.index(limb) + 1) * per_limb,
                dtype=np.int64,
            )
            for limb in limbs
        ]
    )


def statistic_subset_indices(
    feature_dir: Path, sensor_indices: np.ndarray, statistics: list[str] | None
) -> np.ndarray:
    if not statistics:
        return sensor_indices
    names = json.loads((feature_dir / "feature_names.json").read_text(encoding="utf-8"))
    requested = set(statistics)
    selected = np.asarray(
        [index for index in sensor_indices if str(names[int(index)]).rsplit("__", 1)[-1] in requested],
        dtype=np.int64,
    )
    if len(selected) == 0:
        raise ValueError(f"No feature names matched --statistics {sorted(requested)}")
    return selected


def select_subset(features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.asarray(features)[:, indices]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostic participant-grouped CatBoost expert using a fixed sensor subset"
    )
    parser.add_argument("--manifest", default="data/wear/manifest.csv")
    parser.add_argument("--features", default="data/wear/challenge_features")
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds")
    parser.add_argument("--fold-indices", nargs="+", type=int)
    parser.add_argument(
        "--participant-subset",
        nargs="+",
        help="Optional participant universe for nested inner-OOF generation",
    )
    parser.add_argument("--limbs", nargs="+", required=True, choices=LIMB_ORDER)
    parser.add_argument(
        "--statistics",
        nargs="+",
        help="Optional final feature-name components, e.g. skewness kurtosis spectral_entropy",
    )
    parser.add_argument("--augment-left-right", action="store_true")
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
    available_participants = set(frame.participant.astype(str))
    if args.participant_subset:
        requested = {str(value) for value in args.participant_subset}
        unknown = sorted(requested.difference(available_participants))
        if unknown:
            raise ValueError(f"Unknown participants in --participant-subset: {unknown}")
        fold_participant_rows = frame[
            frame.participant.astype(str).isin(requested)
        ].participant.astype(str).tolist()
    else:
        requested = available_participants
        fold_participant_rows = frame.participant.astype(str).tolist()
    folds = load_folds(args.folds, fold_participant_rows, 3)
    selected = (
        [value - 1 for value in args.fold_indices]
        if args.fold_indices
        else list(range(len(folds)))
    )
    if not selected or any(index < 0 or index >= len(folds) for index in selected):
        raise ValueError(f"Fold indices must be between 1 and {len(folds)}")

    participants = sorted(requested, key=participant_sort_key)
    num_classes = int(frame.label.max()) + 1
    oof_probabilities = np.full((len(frame), num_classes), np.nan, dtype=np.float32)
    fold_summaries: list[dict[str, object]] = []

    for fold_index in selected:
        started = time.time()
        test_people = folds[fold_index]
        train_people = [value for value in participants if value not in test_people]
        train_data = [load_participant(feature_dir, value) for value in train_people]
        full_train = np.concatenate([value["features"] for value in train_data])
        train_labels = np.concatenate([value["labels"] for value in train_data])
        indices = statistic_subset_indices(
            feature_dir,
            sensor_subset_indices(full_train.shape[1], args.limbs),
            args.statistics,
        )
        variants = [select_subset(full_train, indices)]
        labels = [train_labels]
        if args.augment_left_right:
            variants.append(select_subset(swap_left_right(full_train), indices))
            labels.append(train_labels)
        train_features = np.concatenate(variants)
        augmented_labels = np.concatenate(labels)
        del full_train, train_data, variants, labels

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
            f"fold {fold_index + 1}: limbs={args.limbs}, train={train_people}, "
            f"test={test_people}, shape={train_features.shape}",
            flush=True,
        )
        model.fit(train_features, augmented_labels)
        del train_features, augmented_labels

        fold_targets: list[np.ndarray] = []
        fold_predictions: list[np.ndarray] = []
        for participant in test_people:
            data = load_participant(feature_dir, participant)
            probabilities = aligned_predict_proba(
                model, select_subset(data["features"], indices), num_classes
            )
            row_indices = data["row_indices"].astype(np.int64)
            oof_probabilities[row_indices] = probabilities
            fold_targets.append(data["labels"])
            fold_predictions.append(probabilities.argmax(axis=1))

        metrics = classification_metrics(
            np.concatenate(fold_targets), np.concatenate(fold_predictions), num_classes
        )
        fold_dir = output / f"fold_{fold_index + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(str(fold_dir / "model.cbm"))
        fold_summary = {
            "fold": fold_index + 1,
            "limbs": args.limbs,
            "augment_left_right": args.augment_left_right,
            "statistics": args.statistics,
            "train_participants": train_people,
            "test_participants": test_people,
            "center_metrics_raw": metrics,
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        save_json(fold_summary, fold_dir / "summary.json")
        fold_summaries.append(fold_summary)

    evaluated = np.isfinite(oof_probabilities).all(axis=1)
    evaluation_frame = frame.loc[evaluated].reset_index(drop=True)
    probabilities = oof_probabilities[evaluated]
    smoothed = smooth_probabilities(
        probabilities,
        evaluation_frame,
        radius_steps=args.smooth_radius,
        sigma_steps=args.smooth_sigma,
    )
    center_metrics = classification_metrics(
        evaluation_frame.label.to_numpy(), smoothed.argmax(axis=1), num_classes
    )
    expanded = expand_probabilities_to_samples(smoothed, evaluation_frame)
    sample_targets, sample_predictions = samplewise_targets(evaluation_frame, expanded)
    sample_metrics = classification_metrics(sample_targets, sample_predictions, num_classes)

    per_participant: dict[str, dict[str, float]] = {}
    participant_values = evaluation_frame.participant.astype(str).to_numpy()
    for participant in sorted(np.unique(participant_values), key=participant_sort_key):
        mask = participant_values == participant
        metrics = classification_metrics(
            evaluation_frame.label.to_numpy()[mask], smoothed[mask].argmax(axis=1), num_classes
        )
        per_participant[participant] = {
            "smoothed_center_macro_f1": float(metrics["macro_f1"])
        }

    summary = {
        "method": "fixed sensor-subset diagnostic expert",
        "limbs": args.limbs,
        "augment_left_right": args.augment_left_right,
        "statistics": args.statistics,
        "participant_subset": args.participant_subset,
        "seed": args.seed,
        "evaluated_folds": [index + 1 for index in selected],
        "folds": fold_summaries,
        "center_metrics_smoothed": center_metrics,
        "samplewise_metrics_smoothed": sample_metrics,
        "per_participant": per_participant,
    }
    save_json(summary, output / "summary.json")
    np.savez_compressed(
        output / "oof_predictions.npz",
        row_indices=np.flatnonzero(evaluated),
        probabilities=probabilities,
        smoothed_probabilities=smoothed,
        targets=evaluation_frame.label.to_numpy(dtype=np.int64),
    )
    print(
        f"smoothed center macro-F1={center_metrics['macro_f1']:.4%}; "
        f"sample-wise={sample_metrics['macro_f1']:.4%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
