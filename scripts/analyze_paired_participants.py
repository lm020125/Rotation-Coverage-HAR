from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_temporal_decoder_cv import evaluate_labels  # noqa: E402
from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.utils import participant_sort_key, save_json  # noqa: E402


def paired_bootstrap_interval(
    differences: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or samples < 1:
        raise ValueError("Bootstrap requires non-empty paired differences and samples")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return tuple(float(value) for value in np.quantile(means, [tail, 1.0 - tail]))


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("Sign-flip testing requires non-empty paired differences")
    observed = abs(float(values.mean()))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        extreme += statistic >= observed - 1e-15
        total += 1
    return extreme / total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Participant-level paired uncertainty for strict WEAR OOF results"
    )
    parser.add_argument("--manifest", default="data/wear/manifest.csv")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-prediction-key", default="gated_predictions")
    parser.add_argument("--candidate-prediction-key", default="gated_predictions")
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = load_manifest(args.manifest).reset_index(drop=True)
    baseline = np.load(Path(args.baseline).resolve() / "oof_predictions.npz")
    candidate = np.load(Path(args.candidate).resolve() / "oof_predictions.npz")
    expected_rows = np.arange(len(frame))
    expected_targets = frame.label.to_numpy(np.int64)
    for name, archive in (("baseline", baseline), ("candidate", candidate)):
        if not np.array_equal(archive["row_indices"], expected_rows):
            raise ValueError(f"{name} OOF rows do not match the manifest")
        if not np.array_equal(archive["targets"], expected_targets):
            raise ValueError(f"{name} OOF targets do not match the manifest")
    baseline_predictions = baseline[args.baseline_prediction_key].astype(np.int64)
    candidate_predictions = candidate[args.candidate_prediction_key].astype(np.int64)
    num_classes = int(frame.label.max()) + 1
    participants = sorted(frame.participant.astype(str).unique(), key=participant_sort_key)
    participant_results: list[dict[str, object]] = []
    sample_differences = []
    center_differences = []
    participant_values = frame.participant.astype(str).to_numpy()
    for participant in participants:
        selected = participant_values == participant
        selected_frame = frame.loc[selected].reset_index(drop=True)
        baseline_metrics = evaluate_labels(
            baseline_predictions[selected], selected_frame, num_classes
        )
        candidate_metrics = evaluate_labels(
            candidate_predictions[selected], selected_frame, num_classes
        )
        sample_difference = (
            candidate_metrics["samplewise"]["macro_f1"]
            - baseline_metrics["samplewise"]["macro_f1"]
        )
        center_difference = (
            candidate_metrics["center"]["macro_f1"]
            - baseline_metrics["center"]["macro_f1"]
        )
        sample_differences.append(sample_difference)
        center_differences.append(center_difference)
        participant_results.append(
            {
                "participant": participant,
                "baseline_samplewise_macro_f1": baseline_metrics["samplewise"][
                    "macro_f1"
                ],
                "candidate_samplewise_macro_f1": candidate_metrics["samplewise"][
                    "macro_f1"
                ],
                "samplewise_difference": sample_difference,
                "baseline_center_macro_f1": baseline_metrics["center"]["macro_f1"],
                "candidate_center_macro_f1": candidate_metrics["center"]["macro_f1"],
                "center_difference": center_difference,
            }
        )

    sample_values = np.asarray(sample_differences)
    center_values = np.asarray(center_differences)
    summary = {
        "unit": "participant",
        "participant_count": len(participants),
        "baseline": args.baseline,
        "candidate": args.candidate,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "samplewise": {
            "mean_paired_difference": float(sample_values.mean()),
            "median_paired_difference": float(np.median(sample_values)),
            "improved_participants": int(np.sum(sample_values > 0.0)),
            "harmed_participants": int(np.sum(sample_values < 0.0)),
            "unchanged_participants": int(np.sum(sample_values == 0.0)),
            "bootstrap_95_ci": paired_bootstrap_interval(
                sample_values, samples=args.bootstrap_samples, seed=args.seed
            ),
            "exact_two_sided_sign_flip_pvalue": exact_sign_flip_pvalue(sample_values),
        },
        "center": {
            "mean_paired_difference": float(center_values.mean()),
            "median_paired_difference": float(np.median(center_values)),
            "improved_participants": int(np.sum(center_values > 0.0)),
            "harmed_participants": int(np.sum(center_values < 0.0)),
            "unchanged_participants": int(np.sum(center_values == 0.0)),
            "bootstrap_95_ci": paired_bootstrap_interval(
                center_values, samples=args.bootstrap_samples, seed=args.seed
            ),
            "exact_two_sided_sign_flip_pvalue": exact_sign_flip_pvalue(center_values),
        },
        "participants": participant_results,
    }
    output = Path(args.output).resolve()
    save_json(summary, output)
    print(
        f"sample mean={summary['samplewise']['mean_paired_difference']:.4%}; "
        f"95% CI={summary['samplewise']['bootstrap_95_ci']}; "
        f"p={summary['samplewise']['exact_two_sided_sign_flip_pvalue']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
