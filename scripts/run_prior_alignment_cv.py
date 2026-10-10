from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_ul_catboost_cv import load_folds, samplewise_targets  # noqa: E402
from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.engine import expand_probabilities_to_samples, smooth_probabilities  # noqa: E402
from rcbd_har.metrics import classification_metrics  # noqa: E402
from rcbd_har.prior_alignment import (  # noqa: E402
    apply_distribution_bias,
    estimate_null_fraction,
    fit_distribution_bias,
    make_shrunk_target_prior,
    mean_participant_prior,
)
from rcbd_har.utils import participant_sort_key, save_json  # noqa: E402


def align_participant(
    probabilities: np.ndarray,
    frame,
    source_prior: np.ndarray,
    estimator: str,
    rho: float,
    beta: float,
    args,
) -> tuple[np.ndarray, float, np.ndarray]:
    estimated_null = estimate_null_fraction(
        probabilities,
        frame,
        estimator,
        smooth_radius=args.smooth_radius,
        smooth_sigma=args.smooth_sigma,
    )
    target_prior = make_shrunk_target_prior(source_prior, estimated_null, rho)
    bias = fit_distribution_bias(probabilities, target_prior)
    return apply_distribution_bias(probabilities, bias, beta), estimated_null, target_prior


def evaluate(probabilities: np.ndarray, frame, num_classes: int, args) -> tuple[dict, np.ndarray]:
    raw = classification_metrics(frame.label.to_numpy(), probabilities.argmax(1), num_classes)
    smoothed = smooth_probabilities(
        probabilities, frame, args.smooth_radius, args.smooth_sigma
    )
    smooth = classification_metrics(frame.label.to_numpy(), smoothed.argmax(1), num_classes)
    expanded = expand_probabilities_to_samples(smoothed, frame)
    sample_targets, sample_predictions = samplewise_targets(frame, expanded)
    sample = classification_metrics(sample_targets, sample_predictions, num_classes)
    return {"raw": raw, "smoothed": smooth, "samplewise": sample}, smoothed


def select_configuration(
    probabilities: np.ndarray,
    frame,
    targets: np.ndarray,
    participant_pool: list[str],
    num_classes: int,
    args,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    participant_values = frame.participant.astype(str).to_numpy()
    cached: dict[tuple[str, float, str], tuple[np.ndarray, np.ndarray]] = {}
    for participant in participant_pool:
        selected = participant_values == participant
        validation_frame = frame.loc[selected].reset_index(drop=True)
        source_people = [value for value in participant_pool if value != participant]
        source_prior = mean_participant_prior(
            targets, participant_values, source_people, num_classes
        )
        for estimator in args.null_estimators:
            estimated_null = estimate_null_fraction(
                probabilities[selected],
                validation_frame,
                estimator,
                smooth_radius=args.smooth_radius,
                smooth_sigma=args.smooth_sigma,
            )
            for rho in args.rhos:
                target_prior = make_shrunk_target_prior(source_prior, estimated_null, rho)
                bias = fit_distribution_bias(probabilities[selected], target_prior)
                cached[(estimator, rho, participant)] = (bias, target_prior)

    scores = []
    for estimator in args.null_estimators:
        for rho in args.rhos:
            for beta in args.betas:
                aligned = np.empty_like(probabilities)
                for participant in participant_pool:
                    selected = participant_values == participant
                    bias, _ = cached[(estimator, rho, participant)]
                    aligned[selected] = apply_distribution_bias(
                        probabilities[selected], bias, beta
                    )
                smoothed = smooth_probabilities(
                    aligned, frame, args.smooth_radius, args.smooth_sigma
                )
                score = classification_metrics(
                    targets, smoothed.argmax(1), num_classes
                )["macro_f1"]
                scores.append(
                    {
                        "null_estimator": estimator,
                        "rho": rho,
                        "beta": beta,
                        "smoothed_center_macro_f1": score,
                    }
                )
    ranking = sorted(
        scores,
        key=lambda value: (
            -float(value["smoothed_center_macro_f1"]),
            value["null_estimator"] != "hard",
            abs(float(value["rho"]) - 1.0),
            abs(float(value["beta"]) - 0.5),
        ),
    )
    return ranking[0], scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested participant-prior alignment for WEAR")
    parser.add_argument("--manifest", default="data/wear/manifest.csv")
    parser.add_argument("--source-output", default="outputs/wear_null_residual_meta_seed0")
    parser.add_argument("--inner-root", default="outputs/wear_null_residual_meta_seed0")
    parser.add_argument("--output", default="outputs/wear_prior_alignment_seed0")
    parser.add_argument("--folds")
    parser.add_argument("--null-estimators", nargs="+", choices=["soft", "hard"], default=["soft", "hard"])
    parser.add_argument("--rhos", nargs="+", type=float, default=[0.0, 0.5, 0.75, 1.0])
    parser.add_argument("--betas", nargs="+", type=float, default=[0.25, 0.4, 0.5, 0.6, 0.75])
    parser.add_argument("--smooth-radius", type=int, default=10)
    parser.add_argument("--smooth-sigma", type=float, default=6.0)
    args = parser.parse_args()

    frame = load_manifest(args.manifest).reset_index(drop=True)
    num_classes = int(frame.label.max()) + 1
    targets = frame.label.to_numpy(np.int64)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_root = Path(args.source_output).resolve()
    inner_root = Path(args.inner_root).resolve()
    folds_path = args.folds or str(source_root / "folds.json")
    folds = load_folds(folds_path, frame.participant.astype(str).tolist(), 3)
    save_json({"test_folds": folds}, output / "folds.json")
    archive = np.load(source_root / "oof_predictions.npz")
    if not np.array_equal(archive["row_indices"], np.arange(len(frame))):
        raise ValueError("Source OOF rows do not match the manifest")
    source_probabilities = archive["refined_probabilities"]
    aligned_oof = np.full_like(source_probabilities, np.nan)
    fold_summaries = []
    all_people = sorted(frame.participant.astype(str).unique(), key=participant_sort_key)

    for fold_index, test_people in enumerate(folds, start=1):
        train_people = [value for value in all_people if value not in test_people]
        inner = np.load(inner_root / f"fold_{fold_index}" / "inner_oof.npz")
        inner_rows = inner["row_indices"].astype(np.int64)
        inner_frame = frame.iloc[inner_rows].reset_index(drop=True)
        inner_probabilities = inner["probabilities"]
        inner_targets = inner["targets"].astype(np.int64)
        if not np.array_equal(inner_targets, inner_frame.label.to_numpy(np.int64)):
            raise ValueError(f"Inner targets do not align in fold {fold_index}")
        selected, meta_scores = select_configuration(
            inner_probabilities,
            inner_frame,
            inner_targets,
            train_people,
            num_classes,
            args,
        )
        source_prior = mean_participant_prior(
            targets,
            frame.participant.astype(str).to_numpy(),
            train_people,
            num_classes,
        )
        participant_details = {}
        for participant in test_people:
            mask = (frame.participant.astype(str) == participant).to_numpy()
            participant_frame = frame.loc[mask].reset_index(drop=True)
            aligned, estimated_null, target_prior = align_participant(
                source_probabilities[mask],
                participant_frame,
                source_prior,
                str(selected["null_estimator"]),
                float(selected["rho"]),
                float(selected["beta"]),
                args,
            )
            aligned_oof[mask] = aligned
            participant_details[participant] = {
                "estimated_null": estimated_null,
                "target_prior": target_prior.tolist(),
            }
        fold_mask = frame.participant.astype(str).isin(test_people).to_numpy()
        fold_frame = frame.loc[fold_mask].reset_index(drop=True)
        base_smoothed = smooth_probabilities(
            source_probabilities[fold_mask], fold_frame, args.smooth_radius, args.smooth_sigma
        )
        aligned_smoothed = smooth_probabilities(
            aligned_oof[fold_mask], fold_frame, args.smooth_radius, args.smooth_sigma
        )
        fold_summary = {
            "fold": fold_index,
            "train_participants": train_people,
            "test_participants": test_people,
            "selected_configuration": selected,
            "meta_selection_scores": meta_scores,
            "participant_details": participant_details,
            "base_smoothed_center_macro_f1": classification_metrics(
                targets[fold_mask], base_smoothed.argmax(1), num_classes
            )["macro_f1"],
            "aligned_smoothed_center_macro_f1": classification_metrics(
                targets[fold_mask], aligned_smoothed.argmax(1), num_classes
            )["macro_f1"],
        }
        fold_dir = output / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_json(fold_summary, fold_dir / "summary.json")
        fold_summaries.append(fold_summary)
        print(
            f"fold {fold_index}: selected={selected}; base={fold_summary['base_smoothed_center_macro_f1']:.4%}; "
            f"aligned={fold_summary['aligned_smoothed_center_macro_f1']:.4%}",
            flush=True,
        )

    if not np.isfinite(aligned_oof).all():
        raise RuntimeError("Aligned OOF predictions are incomplete")
    base_result, base_smoothed = evaluate(source_probabilities, frame, num_classes, args)
    aligned_result, aligned_smoothed = evaluate(aligned_oof, frame, num_classes, args)
    summary = {
        "method": "Nested participant-adaptive prior alignment",
        "transductive": True,
        "strict_outer_test_label_free": True,
        "source_output": str(source_root),
        "folds": fold_summaries,
        "base": base_result,
        "aligned": aligned_result,
    }
    save_json(summary, output / "summary.json")
    np.savez_compressed(
        output / "oof_predictions.npz",
        row_indices=np.arange(len(frame)),
        source_probabilities=source_probabilities,
        aligned_probabilities=aligned_oof,
        aligned_smoothed_probabilities=aligned_smoothed,
        targets=targets,
    )
    print(
        f"base smooth={base_result['smoothed']['macro_f1']:.4%}; "
        f"aligned smooth={aligned_result['smoothed']['macro_f1']:.4%}; "
        f"aligned sample={aligned_result['samplewise']['macro_f1']:.4%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
