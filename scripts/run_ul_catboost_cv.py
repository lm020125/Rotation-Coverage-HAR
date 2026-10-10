from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.engine import expand_probabilities_to_samples, smooth_probabilities  # noqa: E402
from rcbd_har.metrics import classification_metrics  # noqa: E402
from rcbd_har.ul_pairing import (  # noqa: E402
    average_ul_probabilities,
    make_grouped_folds,
    stack_ul_training_variants,
)
from rcbd_har.utils import participant_sort_key, save_json  # noqa: E402


def load_folds(path: str | None, participants_by_row: list[str], n_splits: int) -> list[list[str]]:
    participants = sorted(set(participants_by_row), key=participant_sort_key)
    if path is None:
        folds = make_grouped_folds(participants_by_row, n_splits=n_splits)
    else:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        folds = payload["test_folds"] if isinstance(payload, dict) else payload
        folds = [[str(value) for value in fold] for fold in folds]
    flattened = [value for fold in folds for value in fold]
    if sorted(flattened, key=participant_sort_key) != participants:
        raise ValueError("Fold definition must contain every participant exactly once")
    return folds


def load_participant(feature_dir: Path, participant: str) -> dict[str, np.ndarray]:
    path = feature_dir / f"{participant}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing extracted features: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def aligned_predict_proba(model, values: np.ndarray, num_classes: int) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(values), dtype=np.float32)
    output = np.zeros((len(values), num_classes), dtype=np.float32)
    output[:, np.asarray(model.classes_, dtype=int)] = probabilities
    return output


def samplewise_targets(frame, expanded: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    recording_column = "recording_id" if "recording_id" in frame.columns else "session"
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for group_key, indices in frame.groupby(["participant", recording_column], sort=False).indices.items():
        name = "__".join(str(value) for value in group_key)
        rows = frame.iloc[np.asarray(indices, dtype=int)]
        label_paths = rows.labels_path.drop_duplicates().tolist()
        if len(label_paths) != 1:
            raise ValueError(f"Recording {group_key} maps to multiple label arrays")
        truth = np.load(str(label_paths[0]), allow_pickle=False).astype(np.int64)
        probabilities = expanded[name]
        if len(truth) != len(probabilities):
            raise ValueError(f"Sample count mismatch for {group_key}: {len(truth)} vs {len(probabilities)}")
        targets.append(truth)
        predictions.append(probabilities.argmax(axis=1))
    return np.concatenate(targets), np.concatenate(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-fold WEAR UL CatBoost reproduction")
    parser.add_argument("--manifest", default="data/wear/manifest.csv")
    parser.add_argument("--features", default="data/wear/challenge_features")
    parser.add_argument("--output", default="outputs/wear_ul_catboost_seed0")
    parser.add_argument("--folds")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--fold-indices", nargs="+", type=int)
    # The paper keeps all unlisted CatBoost parameters at their defaults.
    # CatBoost's default random_seed is 0, so use 0 for the reproduction run.
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
    folds = load_folds(
        args.folds, frame.participant.astype(str).tolist(), n_splits=args.n_splits
    )
    selected = (
        [value - 1 for value in args.fold_indices]
        if args.fold_indices
        else list(range(len(folds)))
    )
    if not selected or any(index < 0 or index >= len(folds) for index in selected):
        raise ValueError(f"Fold indices must be between 1 and {len(folds)}")
    save_json({"test_folds": folds}, output / "folds.json")

    num_classes = int(frame.label.max()) + 1
    oof_probabilities = np.full((len(frame), num_classes), np.nan, dtype=np.float32)
    fold_summaries: list[dict[str, object]] = []
    all_participants = sorted(
        frame.participant.astype(str).unique().tolist(), key=participant_sort_key
    )

    for fold_index in selected:
        started = time.time()
        test_participants = folds[fold_index]
        train_participants = [value for value in all_participants if value not in test_participants]
        train_data = [load_participant(feature_dir, value) for value in train_participants]
        train_features = np.concatenate([value["features"] for value in train_data], axis=0)
        train_labels = np.concatenate([value["labels"] for value in train_data], axis=0)
        augmented_features, augmented_labels = stack_ul_training_variants(
            train_features, train_labels
        )
        del train_features, train_data
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
            f"fold {fold_index + 1}/{len(folds)}: train={train_participants}, "
            f"test={test_participants}, augmented={augmented_features.shape}",
            flush=True,
        )
        model.fit(augmented_features, augmented_labels)
        del augmented_features, augmented_labels

        fold_targets: list[np.ndarray] = []
        fold_predictions: list[np.ndarray] = []
        for participant in test_participants:
            data = load_participant(feature_dir, participant)
            probabilities = average_ul_probabilities(
                lambda values: aligned_predict_proba(model, values, num_classes),
                data["features"],
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
        summary = {
            "fold": fold_index + 1,
            "train_participants": train_participants,
            "test_participants": test_participants,
            "center_metrics_raw": metrics,
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        save_json(summary, fold_dir / "summary.json")
        fold_summaries.append(summary)
        print(
            f"fold {fold_index + 1} raw center macro-F1={metrics['macro_f1']:.4%}, "
            f"elapsed={summary['elapsed_minutes']:.1f} min",
            flush=True,
        )

    evaluated = np.isfinite(oof_probabilities).all(axis=1)
    evaluation_frame = frame.loc[evaluated].reset_index(drop=True)
    probabilities = oof_probabilities[evaluated]
    raw_predictions = probabilities.argmax(axis=1)
    raw_metrics = classification_metrics(
        evaluation_frame.label.to_numpy(), raw_predictions, num_classes
    )
    smoothed = smooth_probabilities(
        probabilities,
        evaluation_frame,
        radius_steps=args.smooth_radius,
        sigma_steps=args.smooth_sigma,
    )
    smoothed_metrics = classification_metrics(
        evaluation_frame.label.to_numpy(), smoothed.argmax(axis=1), num_classes
    )
    expanded = expand_probabilities_to_samples(smoothed, evaluation_frame)
    sample_targets, sample_predictions = samplewise_targets(evaluation_frame, expanded)
    sample_metrics = classification_metrics(sample_targets, sample_predictions, num_classes)

    per_participant: dict[str, dict[str, float]] = {}
    for participant in sorted(
        evaluation_frame.participant.unique().tolist(), key=participant_sort_key
    ):
        mask = (evaluation_frame.participant == participant).to_numpy()
        participant_metrics = classification_metrics(
            evaluation_frame.loc[mask, "label"].to_numpy(),
            smoothed[mask].argmax(axis=1),
            num_classes,
        )
        per_participant[participant] = {
            "smoothed_center_macro_f1": float(participant_metrics["macro_f1"])
        }

    summary = {
        "method": "UL CatBoost reproduction",
        "seed": args.seed,
        "paper_parameters": {
            "iterations": args.iterations,
            "depth": args.depth,
            "border_count": args.border_count,
            "auto_class_weights": "Balanced",
            "ul_pairings": 4,
            "smoothing_radius_steps": args.smooth_radius,
            "smoothing_sigma_steps": args.smooth_sigma,
        },
        "evaluated_folds": [index + 1 for index in selected],
        "folds": fold_summaries,
        "center_metrics_raw": raw_metrics,
        "center_metrics_smoothed": smoothed_metrics,
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
        f"raw center macro-F1={raw_metrics['macro_f1']:.4%}; "
        f"smoothed center={smoothed_metrics['macro_f1']:.4%}; "
        f"smoothed sample-wise={sample_metrics['macro_f1']:.4%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
