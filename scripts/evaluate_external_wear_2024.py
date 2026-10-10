from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_temporal_decoder_cv import evaluate_labels  # noqa: E402
from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.utils import participant_sort_key, save_json  # noqa: E402


FROZEN_PREDICTION_SHA256 = (
    "05fd1fe1914266a9eb332a36c87f95c7438371c54edecf0c733464070098c691"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_method(predictions: np.ndarray, frame, num_classes: int) -> dict:
    overall = evaluate_labels(predictions, frame, num_classes)
    participant_values = frame.participant.astype(str).to_numpy()
    participants = {}
    for person in sorted(np.unique(participant_values), key=participant_sort_key):
        selected = participant_values == person
        participants[person] = evaluate_labels(
            predictions[selected], frame.loc[selected].reset_index(drop=True), num_classes
        )
    return {"overall": overall, "participants": participants}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-time evaluation of frozen WEAR 2024 external predictions"
    )
    parser.add_argument(
        "--manifest", default="data/wear/external_2024/labeled_eval/manifest.csv"
    )
    parser.add_argument(
        "--predictions", default="outputs/wear_external_2024/predictions_blind.npz"
    )
    parser.add_argument(
        "--output", default="outputs/wear_external_2024/external_results.json"
    )
    parser.add_argument(
        "--nag-seed-stability",
        help="Optional post-reveal, label-free NAG seed-stability prediction archive",
    )
    args = parser.parse_args()

    prediction_path = Path(args.predictions).resolve()
    actual_hash = sha256_file(prediction_path)
    if actual_hash != FROZEN_PREDICTION_SHA256:
        raise RuntimeError(
            f"Prediction hash changed: {actual_hash} != {FROZEN_PREDICTION_SHA256}"
        )
    frame = load_manifest(args.manifest).reset_index(drop=True)
    people = sorted(frame.participant.astype(str).unique(), key=participant_sort_key)
    if people != ["sbj_18", "sbj_19", "sbj_20", "sbj_21"]:
        raise ValueError(f"Unexpected revealed participants: {people}")
    if np.any(frame.label.to_numpy(np.int64) < 0):
        raise ValueError("Revealed manifest still contains unlabeled rows")

    # The frozen archive stores the pandas participant column as a trusted local
    # object array. Its exact SHA-256 is verified above before pickle loading.
    with np.load(prediction_path, allow_pickle=True) as archive:
        if not np.array_equal(
            archive["row_indices"], np.arange(len(frame), dtype=np.int64)
        ):
            raise ValueError("Frozen prediction rows do not match revealed manifest")
        if not np.array_equal(
            archive["centers"], frame.center.to_numpy(dtype=np.int64)
        ):
            raise ValueError("Frozen prediction centers do not match revealed manifest")
        if not np.array_equal(
            archive["participants"].astype(str), frame.participant.astype(str).to_numpy()
        ):
            raise ValueError("Frozen prediction participants do not match revealed manifest")
        main = archive["main_predictions"].astype(np.int64)
        nag = archive["nag_predictions"].astype(np.int64)
        fold_main = archive["fold_main_predictions"].astype(np.int64)
        fold_nag = archive["fold_nag_predictions"].astype(np.int64)

    num_classes = max(18, int(frame.label.max())) + 1
    main_result = evaluate_method(main, frame, num_classes)
    nag_result = evaluate_method(nag, frame, num_classes)
    fold_results = []
    for fold_index in range(fold_main.shape[0]):
        fold_results.append(
            {
                "fold": fold_index + 1,
                "main": evaluate_method(fold_main[fold_index], frame, num_classes),
                "nag": evaluate_method(fold_nag[fold_index], frame, num_classes),
                "main_nag_disagreement": float(
                    np.mean(fold_main[fold_index] != fold_nag[fold_index])
                ),
            }
        )
    main_score = float(main_result["overall"]["samplewise"]["macro_f1"])
    nag_score = float(nag_result["overall"]["samplewise"]["macro_f1"])
    result = {
        "protocol": "one-time post-hash reveal of official sbj_18-sbj_21 labels",
        "prediction_sha256": actual_hash,
        "revealed_manifest_sha256": sha256_file(Path(args.manifest).resolve()),
        "window_count": len(frame),
        "participants": people,
        "main": main_result,
        "nag": nag_result,
        "nag_minus_main_samplewise_macro_f1": nag_score - main_score,
        "retain_nag_under_frozen_rule": nag_score >= main_score,
        "final_main_nag_disagreement": float(np.mean(main != nag)),
        "fold_model_results": fold_results,
    }
    if args.nag_seed_stability:
        stability_path = Path(args.nag_seed_stability).resolve()
        with np.load(stability_path, allow_pickle=False) as archive:
            if not np.array_equal(
                archive["row_indices"], np.arange(len(frame), dtype=np.int64)
            ):
                raise ValueError("NAG seed-stability rows do not match revealed manifest")
            stability_seeds = archive["seeds"].astype(np.int64)
            stability_predictions = archive["seed_predictions"].astype(np.int64)
        stability_results = []
        stability_scores = []
        for seed, predictions in zip(
            stability_seeds, stability_predictions, strict=True
        ):
            evaluated = evaluate_method(predictions, frame, num_classes)
            score = float(evaluated["overall"]["samplewise"]["macro_f1"])
            stability_scores.append(score)
            stability_results.append(
                {"seed": int(seed), "samplewise_macro_f1": score, "result": evaluated}
            )
        result["nag_seed_stability"] = {
            "prediction_sha256": sha256_file(stability_path),
            "results": stability_results,
            "mean_samplewise_macro_f1": float(np.mean(stability_scores)),
            "std_samplewise_macro_f1": float(np.std(stability_scores)),
            "all_deltas_from_main": [score - main_score for score in stability_scores],
        }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(result, output)
    print(
        f"main samplewise={main_score:.6%}; NAG samplewise={nag_score:.6%}; "
        f"delta={(nag_score - main_score):+.6%}; retain={nag_score >= main_score}",
        flush=True,
    )


if __name__ == "__main__":
    main()
