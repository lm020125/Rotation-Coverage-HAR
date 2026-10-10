from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_prior_alignment_cv import align_participant  # noqa: E402
from run_temporal_decoder_cv import evaluate_labels  # noqa: E402
from run_ul_catboost_cv import load_folds  # noqa: E402
from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.metrics import classification_metrics  # noqa: E402
from rcbd_har.prior_alignment import mean_participant_prior  # noqa: E402
from rcbd_har.protocol_decoder import protocol_constrained_decode  # noqa: E402
from rcbd_har.temporal_decoder import estimate_transition_matrix  # noqa: E402
from rcbd_har.utils import participant_sort_key, save_json  # noqa: E402


def minimum_count_vector(num_classes: int, nonnull_minimum: int) -> np.ndarray:
    bounds = np.full(num_classes, int(nonnull_minimum), dtype=np.int64)
    bounds[0] = 0
    return bounds


def select_configuration(
    probabilities: np.ndarray,
    frame,
    targets: np.ndarray,
    participants: list[str],
    prior_configuration: dict[str, object],
    num_classes: int,
    args,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    participant_values = frame.participant.astype(str).to_numpy()
    configurations = [
        (float(strength), int(minimum_count))
        for strength in args.strengths
        for minimum_count in args.minimum_counts
    ]
    predictions = {
        configuration: np.empty(len(frame), dtype=np.int64)
        for configuration in configurations
    }
    for participant in participants:
        validation = participant_values == participant
        source_people = [value for value in participants if value != participant]
        source = ~validation
        source_prior = mean_participant_prior(
            targets, participant_values, source_people, num_classes
        )
        validation_frame = frame.loc[validation].reset_index(drop=True)
        aligned, _, _ = align_participant(
            probabilities[validation],
            validation_frame,
            source_prior,
            str(prior_configuration["null_estimator"]),
            float(prior_configuration["rho"]),
            float(prior_configuration["beta"]),
            args,
        )
        transition = estimate_transition_matrix(
            targets[source],
            frame.loc[source].reset_index(drop=True),
            num_classes,
            args.laplace,
        )
        for strength, minimum_count in configurations:
            result = protocol_constrained_decode(
                aligned,
                validation_frame,
                transition,
                strength,
                minimum_count_vector(num_classes, minimum_count),
            )
            predictions[(strength, minimum_count)][validation] = result.predictions

    scores: list[dict[str, float | int]] = []
    for strength, minimum_count in configurations:
        macro_f1 = classification_metrics(
            targets, predictions[(strength, minimum_count)], num_classes
        )["macro_f1"]
        scores.append(
            {
                "strength": strength,
                "minimum_count": minimum_count,
                "inner_center_macro_f1": float(macro_f1),
            }
        )
    selected = sorted(
        scores,
        key=lambda value: (
            -float(value["inner_center_macro_f1"]),
            int(value["minimum_count"]),
            abs(float(value["strength"]) - 1.0),
        ),
    )[0]
    return selected, scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nested protocol-constrained dual Viterbi decoding for WEAR"
    )
    parser.add_argument("--manifest", default="data/wear/manifest.csv")
    parser.add_argument("--source-output", default="outputs/wear_prior_alignment_seed0")
    parser.add_argument("--inner-root", default="outputs/wear_null_residual_meta_seed0")
    parser.add_argument("--output", default="outputs/wear_protocol_decoder_seed0")
    parser.add_argument("--folds")
    parser.add_argument(
        "--strengths", nargs="+", type=float, default=[1.0, 1.5, 2.0, 2.5, 3.0]
    )
    parser.add_argument(
        "--minimum-counts", nargs="+", type=int, default=[0, 100, 125, 150, 175]
    )
    parser.add_argument("--laplace", type=float, default=1e-2)
    parser.add_argument("--smooth-radius", type=int, default=10)
    parser.add_argument("--smooth-sigma", type=float, default=6.0)
    args = parser.parse_args()
    if any(value < 0 for value in args.minimum_counts):
        raise ValueError("minimum-counts must be non-negative")

    frame = load_manifest(args.manifest).reset_index(drop=True)
    targets = frame.label.to_numpy(np.int64)
    num_classes = int(targets.max()) + 1
    source_root = Path(args.source_output).resolve()
    inner_root = Path(args.inner_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    folds = load_folds(
        args.folds or str(source_root / "folds.json"),
        frame.participant.astype(str).tolist(),
        3,
    )
    save_json({"test_folds": folds}, output / "folds.json")
    source_archive = np.load(source_root / "oof_predictions.npz")
    if not np.array_equal(source_archive["row_indices"], np.arange(len(frame))):
        raise ValueError("Source OOF rows do not match the manifest")
    probabilities = source_archive["aligned_probabilities"]
    decoded_oof = np.full(len(frame), -1, dtype=np.int64)
    people = sorted(frame.participant.astype(str).unique(), key=participant_sort_key)
    fold_summaries = []

    for fold_index, test_people in enumerate(folds, start=1):
        train_people = [value for value in people if value not in test_people]
        with (source_root / f"fold_{fold_index}" / "summary.json").open(
            encoding="utf-8"
        ) as handle:
            prior_summary = json.load(handle)
        prior_configuration = prior_summary["selected_configuration"]
        inner = np.load(inner_root / f"fold_{fold_index}" / "inner_oof.npz")
        inner_rows = inner["row_indices"].astype(np.int64)
        inner_frame = frame.iloc[inner_rows].reset_index(drop=True)
        inner_targets = inner["targets"].astype(np.int64)
        if not np.array_equal(inner_targets, inner_frame.label.to_numpy(np.int64)):
            raise ValueError(f"Inner targets do not align in fold {fold_index}")
        selected, meta_scores = select_configuration(
            inner["probabilities"],
            inner_frame,
            inner_targets,
            train_people,
            prior_configuration,
            num_classes,
            args,
        )

        train_mask = frame.participant.astype(str).isin(train_people).to_numpy()
        transition = estimate_transition_matrix(
            targets[train_mask],
            frame.loc[train_mask].reset_index(drop=True),
            num_classes,
            args.laplace,
        )
        participant_diagnostics = {}
        for participant in test_people:
            participant_mask = (frame.participant.astype(str) == participant).to_numpy()
            participant_frame = frame.loc[participant_mask].reset_index(drop=True)
            result = protocol_constrained_decode(
                probabilities[participant_mask],
                participant_frame,
                transition,
                float(selected["strength"]),
                minimum_count_vector(num_classes, int(selected["minimum_count"])),
            )
            decoded_oof[participant_mask] = result.predictions
            participant_diagnostics[participant] = {
                "class_biases": result.class_biases.tolist(),
                "class_counts": result.class_counts.tolist(),
                "constraints_satisfied": result.constraints_satisfied,
            }

        test_mask = frame.participant.astype(str).isin(test_people).to_numpy()
        test_frame = frame.loc[test_mask].reset_index(drop=True)
        fold_result = evaluate_labels(decoded_oof[test_mask], test_frame, num_classes)
        center_stride = float(np.median(np.diff(test_frame.center.unique())))
        sample_rate = float(test_frame.sample_rate.iloc[0])
        fold_summary = {
            "fold": fold_index,
            "train_participants": train_people,
            "test_participants": test_people,
            "prior_configuration": prior_configuration,
            "selected_configuration": selected,
            "selected_minimum_seconds": (
                float(selected["minimum_count"]) * center_stride / sample_rate
            ),
            "meta_selection_scores": meta_scores,
            "participant_diagnostics": participant_diagnostics,
            "test": fold_result,
        }
        fold_dir = output / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_json(fold_summary, fold_dir / "summary.json")
        np.save(fold_dir / "transition_matrix.npy", transition)
        fold_summaries.append(fold_summary)
        print(
            f"fold {fold_index}: strength={selected['strength']:g}, "
            f"minimum={selected['minimum_count']}; "
            f"center={fold_result['center']['macro_f1']:.4%}; "
            f"sample={fold_result['samplewise']['macro_f1']:.4%}",
            flush=True,
        )

    if np.any(decoded_oof < 0):
        raise RuntimeError("Decoded OOF predictions are incomplete")
    overall = evaluate_labels(decoded_oof, frame, num_classes)
    summary = {
        "method": "Nested protocol-constrained dual Viterbi decoding",
        "strict_outer_test_label_free": True,
        "transition_source": "outer-training labels only",
        "configuration_selection": "leave-one-training-participant-out inner OOF",
        "constraint_source": "unlabeled target prediction counts and inner-selected bounds",
        "source_output": str(source_root),
        "folds": fold_summaries,
        "overall": overall,
    }
    save_json(summary, output / "summary.json")
    np.savez_compressed(
        output / "oof_predictions.npz",
        row_indices=np.arange(len(frame)),
        probabilities=probabilities,
        decoded_predictions=decoded_oof,
        targets=targets,
    )
    print(
        f"overall center={overall['center']['macro_f1']:.4%}; "
        f"sample={overall['samplewise']['macro_f1']:.4%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
