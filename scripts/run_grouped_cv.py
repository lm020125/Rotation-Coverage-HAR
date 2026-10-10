from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.config import load_config  # noqa: E402
from rcbd_har.experiment import run_group_fold  # noqa: E402
from rcbd_har.metrics import classification_metrics_from_confusion  # noqa: E402
from rcbd_har.utils import participant_sort_key, save_json  # noqa: E402


def make_folds(participants: list[str], n_splits: int) -> list[list[str]]:
    splitter = KFold(n_splits=n_splits, shuffle=False)
    values = np.asarray(sorted(participants, key=participant_sort_key))
    return [values[test].tolist() for _, test in splitter.split(values)]


def load_folds(path: str | None, participants: list[str], n_splits: int) -> list[list[str]]:
    if path is None:
        return make_folds(participants, n_splits)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    folds = payload["test_folds"] if isinstance(payload, dict) else payload
    folds = [[str(value) for value in fold] for fold in folds]
    flattened = [value for fold in folds for value in fold]
    if sorted(flattened) != sorted(participants):
        raise ValueError("Fold file must contain every participant exactly once")
    return folds


def aggregate_fold_metrics(fold_metrics: list[dict[str, object]]) -> dict[str, object]:
    """Pool held-out predictions through summed confusion matrices."""
    confusion = np.sum(
        [np.asarray(item["confusion_matrix"], dtype=np.int64) for item in fold_metrics],
        axis=0,
    )
    aggregate = classification_metrics_from_confusion(confusion)
    raw_confusion = (
        np.sum(
            [np.asarray(item["raw_confusion_matrix"], dtype=np.int64) for item in fold_metrics],
            axis=0,
        )
        if all("raw_confusion_matrix" in item for item in fold_metrics)
        else None
    )
    if raw_confusion is not None:
        aggregate["raw"] = classification_metrics_from_confusion(raw_confusion)
    if all("samplewise" in item for item in fold_metrics):
        samplewise_confusion = np.sum(
            [
                np.asarray(item["samplewise"]["confusion_matrix"], dtype=np.int64)
                for item in fold_metrics
            ],
            axis=0,
        )
        aggregate["samplewise"] = classification_metrics_from_confusion(
            samplewise_confusion
        )
        per_participant: dict[str, object] = {}
        for item in fold_metrics:
            for participant, participant_metrics in item["samplewise"][
                "per_participant"
            ].items():
                if participant in per_participant:
                    raise ValueError(f"Participant {participant} appears in multiple test folds")
                per_participant[participant] = participant_metrics
        aggregate["samplewise"]["per_participant"] = per_participant
    return aggregate


def load_completed_fold(
    output_dir: Path, test_participants: list[str], val_participants: list[str]
) -> dict[str, object] | None:
    """Load a completed fold only when its saved split matches the current protocol."""
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    split_path = output_dir / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Cannot resume {output_dir}: metrics.json exists but split.json is missing"
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    saved_test = [str(value) for value in split.get("test_participants", [])]
    saved_val = [
        str(value)
        for value in split.get(
            "val_participants_for_epoch_selection",
            split.get("val_participants", []),
        )
    ]
    if saved_test != list(test_participants) or saved_val != list(val_participants):
        raise ValueError(
            f"Cannot resume {output_dir}: saved split test={saved_test}, val={saved_val} "
            f"does not match requested test={test_participants}, val={val_participants}"
        )
    metric_test = [str(value) for value in metrics.get("test_participants", [])]
    metric_val = [str(value) for value in metrics.get("val_participants", [])]
    if metric_test != saved_test or metric_val != saved_val:
        raise ValueError(
            f"Cannot resume {output_dir}: metrics participant metadata disagrees with split.json"
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Grouped participant CV for four-limb RCBD-HAR")
    parser.add_argument("--config", default="configs/wear_rcbd.yaml")
    parser.add_argument("--output", default="outputs/wear_grouped_cv")
    parser.add_argument("--folds", help='Optional JSON: {"test_folds": [["P01", ...], ...]}')
    parser.add_argument("--device")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--fold-indices",
        nargs="+",
        type=int,
        help="Optional 1-based fold indices for a fast pilot (for example: 1).",
    )
    parser.add_argument(
        "--skip-refit",
        action="store_true",
        help="Skip full-train refitting after epoch selection; intended for pilot ablations.",
    )
    parser.add_argument(
        "--resume-completed",
        action="store_true",
        help=(
            "Reuse folds that already contain metrics.json and a matching split.json; "
            "mismatched saved protocols abort instead of being silently reused."
        ),
    )
    parser.add_argument(
        "--variant",
        choices=["rcbd", "qmbd", "concat", "ordered_diff", "absdiff", "mean", "deepsets"],
    )
    parser.add_argument("--disable-orientation", action="store_true")
    parser.add_argument("--disable-orientation-consistency", action="store_true")
    parser.add_argument("--single-scale", type=float)
    parser.add_argument("--disable-smoothing", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.variant:
        config["model"]["variant"] = args.variant
    if args.disable_orientation:
        config["model"]["orientation_enabled"] = False
    if args.disable_orientation or args.disable_orientation_consistency:
        config["training"]["orientation_augmentation_probability"] = 0.0
        for key in (
            "orientation_entropy",
            "orientation_information",
            "orientation_equivariance",
            "orientation_canonical_consistency",
            "orientation_prediction_consistency",
            "orientation_embedding_consistency",
        ):
            config["loss"][key] = 0.0
    if args.single_scale is not None:
        config["model"]["scales_seconds"] = [args.single_scale]
    if args.disable_smoothing:
        config.setdefault("evaluation", {})["enabled"] = False
    frame = load_manifest(config["data"]["manifest"])
    participants = sorted(
        frame.participant.astype(str).unique().tolist(), key=participant_sort_key
    )
    cv = config.get("cross_validation", {})
    folds = load_folds(args.folds, participants, int(cv.get("n_splits", 3)))
    selected_fold_indices = (
        [value - 1 for value in args.fold_indices]
        if args.fold_indices is not None
        else list(range(len(folds)))
    )
    if not selected_fold_indices or any(
        value < 0 or value >= len(folds) for value in selected_fold_indices
    ):
        raise ValueError(f"Fold indices must be between 1 and {len(folds)}")
    if len(set(selected_fold_indices)) != len(selected_fold_indices):
        raise ValueError("Fold indices must not contain duplicates")
    seeds = args.seeds or [int(value) for value in cv.get("seeds", [config.get("seed", 42)])]
    val_count = int(cv.get("val_participants_per_fold", 2))
    output_root = Path(args.output)
    seed_summaries = []
    for seed in seeds:
        fold_metrics = []
        for fold_index in selected_fold_indices:
            test_people = folds[fold_index]
            remaining = [person for person in participants if person not in test_people]
            offset = (fold_index * val_count) % len(remaining)
            rotated = remaining[offset:] + remaining[:offset]
            val_people = rotated[:val_count]
            fold_config = copy.deepcopy(config)
            fold_config["seed"] = seed
            fold_output = output_root / f"seed_{seed}" / f"fold_{fold_index + 1}"
            metrics = (
                load_completed_fold(fold_output, test_people, val_people)
                if args.resume_completed
                else None
            )
            if metrics is None:
                metrics = run_group_fold(
                    fold_config,
                    test_people,
                    val_people,
                    fold_output,
                    args.device,
                    refit_full_train=(
                        bool(cv.get("refit_full_train", True)) and not args.skip_refit
                    ),
                )
            else:
                print(f"Reusing completed seed={seed}, fold={fold_index + 1}: {fold_output}")
            fold_metrics.append(metrics)
        aggregate = aggregate_fold_metrics(fold_metrics)
        aggregate["seed"] = seed
        aggregate["evaluated_fold_indices"] = [value + 1 for value in selected_fold_indices]
        aggregate["fold_macro_f1"] = [item["macro_f1"] for item in fold_metrics]
        save_json(aggregate, output_root / f"seed_{seed}" / "summary.json")
        seed_summaries.append(aggregate)
    scores = np.asarray([item["macro_f1"] for item in seed_summaries])
    samplewise_scores = np.asarray(
        [item["samplewise"]["macro_f1"] for item in seed_summaries if "samplewise" in item]
    )
    summary = {
        "seed_macro_f1": scores.tolist(),
        "macro_f1_mean": float(scores.mean()),
        "macro_f1_std": float(scores.std()),
        "folds": folds,
        "evaluated_fold_indices": [value + 1 for value in selected_fold_indices],
        "partial_cv": len(selected_fold_indices) != len(folds),
        "refit_full_train": bool(cv.get("refit_full_train", True)) and not args.skip_refit,
        "seeds": seeds,
        "variant": str(config["model"]["variant"]),
    }
    if len(samplewise_scores) == len(seed_summaries):
        summary.update(
            {
                "seed_samplewise_macro_f1": samplewise_scores.tolist(),
                "samplewise_macro_f1_mean": float(samplewise_scores.mean()),
                "samplewise_macro_f1_std": float(samplewise_scores.std()),
            }
        )
    save_json(summary, output_root / "summary.json")
    label = "Partial-fold" if summary["partial_cv"] else "OOF"
    message = (
        f"{label} center macro-F1={summary['macro_f1_mean']:.4f} "
        f"+/- {summary['macro_f1_std']:.4f}"
    )
    if "samplewise_macro_f1_mean" in summary:
        message += (
            f"; sample-wise={summary['samplewise_macro_f1_mean']:.4f} "
            f"+/- {summary['samplewise_macro_f1_std']:.4f}"
        )
    print(message)


if __name__ == "__main__":
    main()
