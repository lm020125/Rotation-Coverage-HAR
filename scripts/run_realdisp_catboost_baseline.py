from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.engine import expand_probabilities_to_samples, smooth_probabilities  # noqa: E402
from rcbd_har.metrics import classification_metrics  # noqa: E402
from rcbd_har.ul_pairing import (  # noqa: E402
    average_ul_probabilities,
    stack_ul_training_variants,
    swap_left_right,
)
from rcbd_har.utils import participant_sort_key, save_json  # noqa: E402


DEFAULT_TRAIN = [f"subject{value}" for value in range(1, 11)]
DEFAULT_VAL = ["subject11", "subject12"]
DEFAULT_TEST = [f"subject{value}" for value in range(13, 18)]


def load_participant(feature_dir: Path, participant: str) -> dict[str, np.ndarray]:
    path = feature_dir / f"{participant}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing extracted features: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"features", "labels", "centers", "row_indices"}
        if missing := required.difference(data.files):
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        return {name: np.asarray(data[name]) for name in data.files}


def aligned_predict_proba(model, values: np.ndarray, num_classes: int) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(values), dtype=np.float32)
    output = np.zeros((len(values), num_classes), dtype=np.float32)
    output[:, np.asarray(model.classes_, dtype=int)] = probabilities
    return output


def transform_training(
    features: np.ndarray, labels: np.ndarray, pairing: str
) -> tuple[np.ndarray, np.ndarray]:
    if pairing == "full":
        return features, labels
    if pairing == "lr_swap":
        return np.concatenate([features, swap_left_right(features)]), np.tile(labels, 2)
    if pairing == "ul":
        return stack_ul_training_variants(features, labels)
    raise ValueError(f"Unknown pairing mode: {pairing}")


def predict_with_pairing(
    model, features: np.ndarray, pairing: str, num_classes: int
) -> np.ndarray:
    def predict(values: np.ndarray) -> np.ndarray:
        return aligned_predict_proba(model, values, num_classes)

    if pairing == "full":
        return predict(features)
    if pairing == "lr_swap":
        return 0.5 * (predict(features) + predict(swap_left_right(features)))
    if pairing == "ul":
        return average_ul_probabilities(predict, features)
    raise ValueError(f"Unknown pairing mode: {pairing}")


def collect_features(
    feature_dir: Path, participants: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    bundles = [load_participant(feature_dir, participant) for participant in participants]
    feature_counts = {value["features"].shape[1] for value in bundles}
    if len(feature_counts) != 1:
        raise ValueError(f"Feature dimensions differ: {sorted(feature_counts)}")
    return (
        np.concatenate([value["features"] for value in bundles], axis=0),
        np.concatenate([value["labels"] for value in bundles], axis=0).astype(np.int64),
    )


def samplewise_valid_metrics(
    probabilities: np.ndarray, frame, num_classes: int
) -> dict[str, object]:
    """Nearest-center sample metrics after explicitly excluding REALDISP Null (-1)."""
    expanded = expand_probabilities_to_samples(probabilities, frame)
    recording_column = "recording_id" if "recording_id" in frame.columns else "session"
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    participant_targets: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    participant_predictions: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    for group_key, indices in frame.groupby(
        ["participant", recording_column], sort=False
    ).indices.items():
        rows = frame.iloc[np.asarray(indices, dtype=np.int64)]
        paths = rows.labels_path.drop_duplicates().tolist()
        if len(paths) != 1:
            raise ValueError(f"Recording {group_key} maps to multiple label arrays")
        truth = np.load(str(paths[0]), allow_pickle=False).astype(np.int64)
        name = "__".join(str(value) for value in group_key)
        predicted = expanded[name].argmax(axis=1).astype(np.int64)
        valid = (truth >= 0) & (truth < num_classes)
        truth, predicted = truth[valid], predicted[valid]
        participant = str(group_key[0])
        targets.append(truth)
        predictions.append(predicted)
        participant_targets[participant].append(truth)
        participant_predictions[participant].append(predicted)
    metrics = classification_metrics(
        np.concatenate(targets), np.concatenate(predictions), num_classes
    )
    metrics["null_policy"] = "exclude raw labels outside 0..num_classes-1"
    metrics["per_participant"] = {
        participant: classification_metrics(
            np.concatenate(participant_targets[participant]),
            np.concatenate(participant_predictions[participant]),
            num_classes,
        )
        for participant in sorted(participant_targets, key=participant_sort_key)
    }
    return metrics


def evaluate_scenario(
    model,
    manifest,
    feature_dir: Path,
    participants: list[str],
    pairing: str,
    num_classes: int,
    smooth_radius_steps: int,
    smooth_sigma_steps: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    frames = []
    probabilities = []
    for participant in participants:
        data = load_participant(feature_dir, participant)
        rows = manifest.iloc[data["row_indices"].astype(np.int64)].copy()
        if not np.array_equal(
            rows.label.to_numpy(dtype=np.int64), data["labels"].astype(np.int64)
        ):
            raise ValueError(f"Feature labels do not match manifest for {participant}")
        if set(rows.participant.astype(str)) != {participant}:
            raise ValueError(f"Feature row indices point outside {participant}")
        rows["_manifest_row_index"] = data["row_indices"].astype(np.int64)
        frames.append(rows)
        probabilities.append(
            predict_with_pairing(model, data["features"], pairing, num_classes)
        )
    frame = pd.concat(frames, ignore_index=True)
    raw_probabilities = np.concatenate(probabilities, axis=0)
    smoothed_probabilities = smooth_probabilities(
        raw_probabilities,
        frame,
        radius_steps=smooth_radius_steps,
        sigma_steps=smooth_sigma_steps,
    )
    raw = classification_metrics(
        frame.label.to_numpy(dtype=np.int64), raw_probabilities.argmax(axis=1), num_classes
    )
    smoothed = classification_metrics(
        frame.label.to_numpy(dtype=np.int64),
        smoothed_probabilities.argmax(axis=1),
        num_classes,
    )
    per_participant: dict[str, dict[str, object]] = {}
    for participant in participants:
        mask = (frame.participant.astype(str) == participant).to_numpy()
        per_participant[participant] = {
            "raw": classification_metrics(
                frame.loc[mask, "label"].to_numpy(dtype=np.int64),
                raw_probabilities[mask].argmax(axis=1),
                num_classes,
            ),
            "smoothed": classification_metrics(
                frame.loc[mask, "label"].to_numpy(dtype=np.int64),
                smoothed_probabilities[mask].argmax(axis=1),
                num_classes,
            ),
        }
    result = {
        "participants": participants,
        "center_raw": raw,
        "center_smoothed": smoothed,
        "samplewise_smoothed_valid_activity": samplewise_valid_metrics(
            smoothed_probabilities, frame, num_classes
        ),
        "per_participant": per_participant,
    }
    arrays = {
        "row_indices": frame["_manifest_row_index"].to_numpy(dtype=np.int64),
        "targets": frame.label.to_numpy(dtype=np.int64),
        "probabilities": raw_probabilities,
        "smoothed_probabilities": smoothed_probabilities,
    }
    return result, arrays


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strong REALDISP handcrafted-feature CatBoost baseline under the fixed "
            "subjects 1-10/11-12/13-17 protocol."
        )
    )
    parser.add_argument("--ideal-manifest", required=True)
    parser.add_argument("--self-manifest", required=True)
    parser.add_argument("--ideal-features", required=True)
    parser.add_argument("--self-features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-participants", nargs="+", default=DEFAULT_TRAIN)
    parser.add_argument("--val-participants", nargs="+", default=DEFAULT_VAL)
    parser.add_argument("--test-participants", nargs="+", default=DEFAULT_TEST)
    parser.add_argument("--pairing", choices=["full", "lr_swap", "ul"], default="ul")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--border-count", type=int, default=32)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--task-type", choices=["GPU", "CPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=50)
    parser.add_argument("--smooth-radius-steps", type=int, default=4)
    parser.add_argument("--smooth-sigma-steps", type=float, default=2.0)
    args = parser.parse_args()

    from catboost import CatBoostClassifier

    if args.iterations <= 0 or args.depth <= 0 or args.border_count <= 0:
        raise ValueError("CatBoost iteration, depth, and border parameters must be positive")
    train = [str(value) for value in args.train_participants]
    val = [str(value) for value in args.val_participants]
    test = [str(value) for value in args.test_participants]
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise ValueError("Train, validation, and test participants must be disjoint")

    ideal_manifest = load_manifest(args.ideal_manifest).reset_index(drop=True)
    self_manifest = load_manifest(args.self_manifest).reset_index(drop=True)
    available_ideal = set(ideal_manifest.participant.astype(str).unique())
    available_self = set(self_manifest.participant.astype(str).unique())
    if missing := set(train + val + test).difference(available_ideal):
        raise ValueError(f"Ideal manifest misses participants: {sorted(missing)}")
    if missing := set(test).difference(available_self):
        raise ValueError(f"Self manifest misses test participants: {sorted(missing)}")

    ideal_features = Path(args.ideal_features).resolve()
    self_features = Path(args.self_features).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_features, train_labels = collect_features(ideal_features, train)
    val_features, val_labels = collect_features(ideal_features, val)
    num_classes = int(max(ideal_manifest.label.max(), self_manifest.label.max())) + 1
    fit_features, fit_labels = transform_training(
        train_features, train_labels, args.pairing
    )
    eval_features, eval_labels = transform_training(val_features, val_labels, args.pairing)

    parameters: dict[str, object] = {
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1:average=Macro",
        "iterations": args.iterations,
        "depth": args.depth,
        "border_count": args.border_count,
        "auto_class_weights": "Balanced",
        "random_seed": args.seed,
        "task_type": args.task_type,
        "thread_count": args.thread_count,
        "verbose": args.verbose,
        "allow_writing_files": False,
    }
    if args.task_type == "GPU":
        parameters["devices"] = args.devices
    if args.learning_rate is not None:
        parameters["learning_rate"] = args.learning_rate
    model = CatBoostClassifier(**parameters)
    print(
        f"fit pairing={args.pairing}, train={fit_features.shape}, "
        f"validation={eval_features.shape}, seed={args.seed}",
        flush=True,
    )
    started = time.time()
    model.fit(
        fit_features,
        fit_labels,
        eval_set=(eval_features, eval_labels),
        use_best_model=True,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    elapsed_minutes = (time.time() - started) / 60.0
    model.save_model(str(output / "model.cbm"))

    validation, validation_arrays = evaluate_scenario(
        model,
        ideal_manifest,
        ideal_features,
        val,
        args.pairing,
        num_classes,
        args.smooth_radius_steps,
        args.smooth_sigma_steps,
    )
    ideal_test, ideal_arrays = evaluate_scenario(
        model,
        ideal_manifest,
        ideal_features,
        test,
        args.pairing,
        num_classes,
        args.smooth_radius_steps,
        args.smooth_sigma_steps,
    )
    self_test, self_arrays = evaluate_scenario(
        model,
        self_manifest,
        self_features,
        test,
        args.pairing,
        num_classes,
        args.smooth_radius_steps,
        args.smooth_sigma_steps,
    )
    protocol = {
        "name": "REALDISP fixed user-independent CatBoost baseline",
        "train_scenario": "ideal",
        "evaluation_scenarios": ["ideal", "self"],
        "train_participants": train,
        "validation_participants": val,
        "test_participants": test,
        "test_labels_used_for_model_selection": False,
        "pairing": args.pairing,
        "postprocessing": {
            "type": "Gaussian probability smoothing",
            "radius_steps": args.smooth_radius_steps,
            "sigma_steps": args.smooth_sigma_steps,
        },
        "num_classes": num_classes,
        "primary_metric": "center_smoothed.macro_f1",
        "null_policy": "manifest excludes Null-centered windows",
        "catboost_parameters": parameters,
        "early_stopping_rounds": args.early_stopping_rounds,
        "best_iteration": int(model.get_best_iteration()),
        "elapsed_minutes": elapsed_minutes,
        "manifests": {
            "ideal": str(Path(args.ideal_manifest).resolve()),
            "self": str(Path(args.self_manifest).resolve()),
        },
        "feature_directories": {
            "ideal": str(ideal_features),
            "self": str(self_features),
        },
    }
    summary = {
        "method": f"handcrafted CatBoost ({args.pairing})",
        "seed": args.seed,
        "protocol": protocol,
        "validation_ideal": validation,
        "test_ideal": ideal_test,
        "test_self": self_test,
        "ideal_to_self_macro_f1_drop": float(
            ideal_test["center_smoothed"]["macro_f1"]
            - self_test["center_smoothed"]["macro_f1"]
        ),
    }
    save_json(protocol, output / "protocol.json")
    save_json(summary, output / "summary.json")
    np.savez_compressed(output / "validation_predictions.npz", **validation_arrays)
    np.savez_compressed(output / "ideal_test_predictions.npz", **ideal_arrays)
    np.savez_compressed(output / "self_test_predictions.npz", **self_arrays)
    np.save(
        output / "feature_importance.npy",
        np.asarray(model.get_feature_importance(), dtype=np.float32),
        allow_pickle=False,
    )
    print(
        f"validation Ideal={validation['center_smoothed']['macro_f1']:.4%}; "
        f"test Ideal={ideal_test['center_smoothed']['macro_f1']:.4%}; "
        f"test Self={self_test['center_smoothed']['macro_f1']:.4%}; "
        f"drop={summary['ideal_to_self_macro_f1_drop']:.4%}; "
        f"elapsed={elapsed_minutes:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
