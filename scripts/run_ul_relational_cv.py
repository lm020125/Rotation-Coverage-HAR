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
from rcbd_har.relational_features import make_ul_relational_variant  # noqa: E402
from rcbd_har.ul_pairing import UL_PAIRS, make_grouped_folds  # noqa: E402
from rcbd_har.utils import participant_sort_key, save_json  # noqa: E402


def load_folds(path: str | None, participants: list[str], n_splits: int) -> list[list[str]]:
    unique = sorted(set(participants), key=participant_sort_key)
    if path:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        folds = payload["test_folds"] if isinstance(payload, dict) else payload
    else:
        folds = make_grouped_folds(participants, n_splits=n_splits)
    folds = [[str(value) for value in fold] for fold in folds]
    flattened = [value for fold in folds for value in fold]
    if sorted(flattened, key=participant_sort_key) != unique:
        raise ValueError("Fold definition must contain every participant exactly once")
    return folds


def load_participant(base_dir: Path, relation_dir: Path, participant: str) -> dict[str, np.ndarray]:
    with np.load(base_dir / f"{participant}.npz", allow_pickle=False) as source:
        result = {name: np.asarray(source[name]) for name in source.files}
    with np.load(relation_dir / f"{participant}.npz", allow_pickle=False) as source:
        for name in ("limb_features", "pair_features"):
            result[name] = np.asarray(source[name])
        if not np.array_equal(result["row_indices"], source["row_indices"]):
            raise ValueError(f"Base and relational rows differ for {participant}")
    return result


def aligned_predict_proba(model, values: np.ndarray, num_classes: int) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(values), dtype=np.float32)
    output = np.zeros((len(values), num_classes), dtype=np.float32)
    output[:, np.asarray(model.classes_, dtype=int)] = probabilities
    return output


def variants(data: dict[str, np.ndarray], feature_set: str) -> list[np.ndarray]:
    return [
        make_ul_relational_variant(
            data["features"],
            data["limb_features"],
            data["pair_features"],
            pair_index,
            feature_set=feature_set,
        )
        for pair_index in range(len(UL_PAIRS))
    ]


def samplewise_targets(frame, expanded: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    recording_column = "recording_id" if "recording_id" in frame.columns else "session"
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for key, indices in frame.groupby(["participant", recording_column], sort=False).indices.items():
        rows = frame.iloc[np.asarray(indices, dtype=int)]
        paths = rows.labels_path.drop_duplicates().tolist()
        if len(paths) != 1:
            raise ValueError(f"Recording {key} maps to multiple label arrays")
        truth = np.load(paths[0], allow_pickle=False).astype(np.int64)
        name = "__".join(str(value) for value in key)
        if len(truth) != len(expanded[name]):
            raise ValueError(f"Sample count mismatch for {key}")
        targets.append(truth)
        predictions.append(expanded[name].argmax(axis=1))
    return np.concatenate(targets), np.concatenate(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description="WEAR relational-spectral UL CatBoost CV")
    parser.add_argument("--manifest", default="data/wear/manifest.csv")
    parser.add_argument("--base-features", default="data/wear/challenge_features")
    parser.add_argument("--relational-features", default="data/wear/relational_features")
    parser.add_argument("--output", default="outputs/wear_ul_relational_seed0")
    parser.add_argument("--folds")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--fold-indices", nargs="+", type=int)
    parser.add_argument(
        "--participant-subset",
        nargs="+",
        help="Optional participant subset for nested inner-OOF training",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature-set", choices=["full", "limb", "pair"], default="full")
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
    base_dir = Path(args.base_features).resolve()
    relation_dir = Path(args.relational_features).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    available_participants = set(frame.participant.astype(str))
    if args.participant_subset:
        requested = {str(value) for value in args.participant_subset}
        if missing := requested.difference(available_participants):
            raise ValueError(f"Unknown participant subset: {sorted(missing)}")
        participant_rows = frame.participant.astype(str).isin(requested)
        fold_participants = frame.loc[participant_rows, "participant"].astype(str).tolist()
    else:
        requested = available_participants
        fold_participants = frame.participant.astype(str).tolist()
    folds = load_folds(args.folds, fold_participants, args.n_splits)
    selected = [value - 1 for value in args.fold_indices] if args.fold_indices else list(range(len(folds)))
    if not selected or any(value < 0 or value >= len(folds) for value in selected):
        raise ValueError(f"Fold indices must be between 1 and {len(folds)}")
    save_json({"test_folds": folds}, output / "folds.json")

    participants = sorted(requested, key=participant_sort_key)
    num_classes = int(frame.label.max()) + 1
    oof = np.full((len(frame), num_classes), np.nan, np.float32)
    oof_pairs = np.full((len(frame), len(UL_PAIRS), num_classes), np.nan, np.float32)
    fold_summaries: list[dict[str, object]] = []
    input_features = None

    for fold_index in selected:
        started = time.time()
        test_people = folds[fold_index]
        train_people = [value for value in participants if value not in test_people]
        train_data = [load_participant(base_dir, relation_dir, value) for value in train_people]
        train_variants = [
            variant for data in train_data for variant in variants(data, args.feature_set)
        ]
        augmented_features = np.concatenate(train_variants, axis=0)
        augmented_labels = np.concatenate(
            [data["labels"] for data in train_data for _ in UL_PAIRS], axis=0
        )
        input_features = augmented_features.shape[1]
        del train_variants, train_data
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
            f"augmented={augmented_features.shape}",
            flush=True,
        )
        model.fit(augmented_features, augmented_labels)
        del augmented_features, augmented_labels

        fold_targets = []
        fold_predictions = []
        for participant in test_people:
            data = load_participant(base_dir, relation_dir, participant)
            probabilities_by_pair = np.stack(
                [
                    aligned_predict_proba(model, value, num_classes)
                    for value in variants(data, args.feature_set)
                ],
                axis=1,
            )
            probabilities = probabilities_by_pair.mean(axis=1)
            rows = data["row_indices"].astype(np.int64)
            oof[rows] = probabilities
            oof_pairs[rows] = probabilities_by_pair
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
            "train_participants": train_people,
            "test_participants": test_people,
            "center_metrics_raw": metrics,
            "elapsed_minutes": (time.time() - started) / 60,
        }
        save_json(summary, fold_dir / "summary.json")
        fold_summaries.append(summary)
        print(
            f"fold {fold_index + 1} raw center macro-F1={metrics['macro_f1']:.4%}, "
            f"elapsed={summary['elapsed_minutes']:.1f} min",
            flush=True,
        )

    evaluated = np.isfinite(oof).all(axis=1)
    evaluation_frame = frame.loc[evaluated].reset_index(drop=True)
    probabilities = oof[evaluated]
    raw_metrics = classification_metrics(
        evaluation_frame.label.to_numpy(), probabilities.argmax(axis=1), num_classes
    )
    smoothed = smooth_probabilities(
        probabilities, evaluation_frame, radius_steps=args.smooth_radius, sigma_steps=args.smooth_sigma
    )
    smooth_metrics = classification_metrics(
        evaluation_frame.label.to_numpy(), smoothed.argmax(axis=1), num_classes
    )
    expanded = expand_probabilities_to_samples(smoothed, evaluation_frame)
    sample_targets, sample_predictions = samplewise_targets(evaluation_frame, expanded)
    sample_metrics = classification_metrics(sample_targets, sample_predictions, num_classes)
    summary = {
        "method": "Relational-spectral UL CatBoost",
        "seed": args.seed,
        "participant_subset": args.participant_subset,
        "feature_set": args.feature_set,
        "input_feature_count": input_features,
        "evaluated_folds": [value + 1 for value in selected],
        "folds": fold_summaries,
        "center_metrics_raw": raw_metrics,
        "center_metrics_smoothed": smooth_metrics,
        "samplewise_metrics_smoothed": sample_metrics,
    }
    save_json(summary, output / "summary.json")
    np.savez_compressed(
        output / "oof_predictions.npz",
        row_indices=np.flatnonzero(evaluated),
        probabilities=probabilities,
        pair_probabilities=oof_pairs[evaluated],
        smoothed_probabilities=smoothed,
        targets=evaluation_frame.label.to_numpy(np.int64),
    )
    print(
        f"raw center={raw_metrics['macro_f1']:.4%}; smoothed center={smooth_metrics['macro_f1']:.4%}; "
        f"smoothed sample-wise={sample_metrics['macro_f1']:.4%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
