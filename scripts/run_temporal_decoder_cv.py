from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_prior_alignment_cv import align_participant  # noqa: E402
from run_ul_catboost_cv import load_folds, samplewise_targets  # noqa: E402
from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.engine import expand_probabilities_to_samples  # noqa: E402
from rcbd_har.metrics import classification_metrics  # noqa: E402
from rcbd_har.prior_alignment import mean_participant_prior  # noqa: E402
from rcbd_har.temporal_decoder import (  # noqa: E402
    estimate_transition_matrix,
    viterbi_decode,
)
from rcbd_har.utils import participant_sort_key, save_json  # noqa: E402


def decode_as_probabilities(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes, dtype=np.float32)[np.asarray(labels, dtype=np.int64)]


def evaluate_labels(labels: np.ndarray, frame, num_classes: int) -> dict:
    center = classification_metrics(frame.label.to_numpy(), labels, num_classes)
    one_hot = decode_as_probabilities(labels, num_classes)
    expanded = expand_probabilities_to_samples(one_hot, frame)
    sample_targets, sample_predictions = samplewise_targets(frame, expanded)
    sample = classification_metrics(sample_targets, sample_predictions, num_classes)
    return {"center": center, "samplewise": sample}


def select_strength(
    probabilities: np.ndarray,
    frame,
    targets: np.ndarray,
    participants: list[str],
    configuration: dict[str, object],
    num_classes: int,
    args,
) -> tuple[float, list[dict[str, float]]]:
    participant_values = frame.participant.astype(str).to_numpy()
    predictions = {
        strength: np.empty(len(frame), dtype=np.int64) for strength in args.strengths
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
            str(configuration["null_estimator"]),
            float(configuration["rho"]),
            float(configuration["beta"]),
            args,
        )
        transition = estimate_transition_matrix(
            targets[source], frame.loc[source].reset_index(drop=True), num_classes, args.laplace
        )
        for strength in args.strengths:
            predictions[strength][validation] = viterbi_decode(
                aligned, validation_frame, transition, strength
            )
    scores = [
        {
            "strength": float(strength),
            "inner_center_macro_f1": float(
                classification_metrics(targets, predictions[strength], num_classes)["macro_f1"]
            ),
        }
        for strength in args.strengths
    ]
    best = sorted(
        scores,
        key=lambda value: (
            -value["inner_center_macro_f1"],
            abs(value["strength"] - 0.5),
            value["strength"],
        ),
    )[0]
    return best["strength"], scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested transition decoding for WEAR")
    parser.add_argument("--manifest", default="data/wear/manifest.csv")
    parser.add_argument("--source-output", default="outputs/wear_prior_alignment_seed0")
    parser.add_argument("--inner-root", default="outputs/wear_null_residual_meta_seed0")
    parser.add_argument("--output", default="outputs/wear_temporal_decoder_seed0")
    parser.add_argument("--folds")
    parser.add_argument(
        "--strengths", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--laplace", type=float, default=1e-2)
    parser.add_argument("--smooth-radius", type=int, default=10)
    parser.add_argument("--smooth-sigma", type=float, default=6.0)
    args = parser.parse_args()

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

    import json

    for fold_index, test_people in enumerate(folds, start=1):
        train_people = [value for value in people if value not in test_people]
        with (source_root / f"fold_{fold_index}" / "summary.json").open(
            encoding="utf-8"
        ) as handle:
            prior_summary = json.load(handle)
        configuration = prior_summary["selected_configuration"]
        inner = np.load(inner_root / f"fold_{fold_index}" / "inner_oof.npz")
        inner_rows = inner["row_indices"].astype(np.int64)
        inner_frame = frame.iloc[inner_rows].reset_index(drop=True)
        inner_targets = inner["targets"].astype(np.int64)
        if not np.array_equal(inner_targets, inner_frame.label.to_numpy(np.int64)):
            raise ValueError(f"Inner targets do not align in fold {fold_index}")
        selected_strength, meta_scores = select_strength(
            inner["probabilities"],
            inner_frame,
            inner_targets,
            train_people,
            configuration,
            num_classes,
            args,
        )
        train_mask = frame.participant.astype(str).isin(train_people).to_numpy()
        test_mask = frame.participant.astype(str).isin(test_people).to_numpy()
        transition = estimate_transition_matrix(
            targets[train_mask], frame.loc[train_mask].reset_index(drop=True), num_classes, args.laplace
        )
        test_frame = frame.loc[test_mask].reset_index(drop=True)
        decoded = viterbi_decode(
            probabilities[test_mask], test_frame, transition, selected_strength
        )
        decoded_oof[test_mask] = decoded
        fold_result = evaluate_labels(decoded, test_frame, num_classes)
        fold_summary = {
            "fold": fold_index,
            "train_participants": train_people,
            "test_participants": test_people,
            "prior_configuration": configuration,
            "selected_strength": selected_strength,
            "meta_selection_scores": meta_scores,
            "test": fold_result,
        }
        fold_dir = output / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_json(fold_summary, fold_dir / "summary.json")
        np.save(fold_dir / "transition_matrix.npy", transition)
        fold_summaries.append(fold_summary)
        print(
            f"fold {fold_index}: strength={selected_strength:g}; "
            f"center={fold_result['center']['macro_f1']:.4%}; "
            f"sample={fold_result['samplewise']['macro_f1']:.4%}",
            flush=True,
        )

    if np.any(decoded_oof < 0):
        raise RuntimeError("Decoded OOF predictions are incomplete")
    overall = evaluate_labels(decoded_oof, frame, num_classes)
    summary = {
        "method": "Nested participant-independent transition decoding",
        "strict_outer_test_label_free": True,
        "transition_source": "outer-training labels only",
        "strength_selection": "leave-one-training-participant-out inner OOF",
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
