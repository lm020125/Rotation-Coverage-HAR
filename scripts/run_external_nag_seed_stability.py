from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_external_wear_2024_blind import (  # noqa: E402
    infer_nag_fold,
    ordered_feature_data,
    validate_blind_manifest,
)
from run_null_aware_calibration_cv import load_participant  # noqa: E402
from run_ul_catboost_cv import load_folds  # noqa: E402
from rcbd_har.challenge_features import feature_names  # noqa: E402
from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.participant_calibration import amplitude_feature_indices  # noqa: E402
from rcbd_har.protocol_gate import hard_vote_with_probability_tiebreak  # noqa: E402
from rcbd_har.utils import participant_sort_key, save_json  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label-free NAG external seed-stability predictions"
    )
    parser.add_argument("--dev-manifest", default="data/wear/manifest.csv")
    parser.add_argument("--dev-features", default="data/wear/challenge_features")
    parser.add_argument(
        "--external-manifest", default="data/wear/external_2024/unlabeled/manifest.csv"
    )
    parser.add_argument(
        "--external-features",
        default="data/wear/external_2024/unlabeled/challenge_features",
    )
    parser.add_argument(
        "--main-predictions", default="outputs/wear_external_2024/predictions_blind.npz"
    )
    parser.add_argument("--folds", default="outputs/wear_sa_da_ccfre_focus_seed0/folds.json")
    parser.add_argument("--nag-root", default="outputs/wear_nag_psf15_seed0")
    parser.add_argument("--output", default="outputs/wear_external_2024/nag_seed_stability_blind.npz")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 3407, 114514])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--border-count", type=int, default=32)
    parser.add_argument("--minimum-factor", type=float, default=0.5)
    parser.add_argument("--maximum-factor", type=float, default=2.0)
    parser.add_argument("--task-type", choices=["GPU", "CPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--thread-count", type=int, default=-1)
    args = parser.parse_args()

    dev_frame = load_manifest(args.dev_manifest).reset_index(drop=True)
    external_frame = load_manifest(args.external_manifest).reset_index(drop=True)
    external_people = ["sbj_18", "sbj_19", "sbj_20", "sbj_21"]
    validate_blind_manifest(external_frame, external_people)
    args.num_classes = int(dev_frame.label.max()) + 1
    dev_people = sorted(
        dev_frame.participant.astype(str).unique(), key=participant_sort_key
    )
    folds = load_folds(args.folds, dev_frame.participant.astype(str).tolist(), 3)
    dev_data = {
        person: load_participant(Path(args.dev_features).resolve(), person)
        for person in dev_people
    }
    external_data, _ = ordered_feature_data(
        Path(args.external_features).resolve(), external_people, len(external_frame)
    )
    all_data = {**dev_data, **external_data}
    amplitude_indices = amplitude_feature_indices(feature_names())
    with np.load(args.main_predictions, allow_pickle=True) as archive:
        fold_main = archive["fold_main_predictions"].astype(np.int64)
        support = archive["fold_probability_support"].astype(np.float32)
        frozen_main = archive["main_predictions"].astype(np.int64)

    all_seed_predictions: list[np.ndarray] = []
    all_seed_folds: list[np.ndarray] = []
    diagnostics = []
    for seed in args.seeds:
        args.seed = int(seed)
        seed_folds: list[np.ndarray] = []
        seed_details = []
        for fold_index, test_people in enumerate(folds, start=1):
            train_people = [person for person in dev_people if person not in test_people]
            prediction, detail = infer_nag_fold(
                fold_index=fold_index,
                train_people=train_people,
                external_people=external_people,
                all_feature_data=all_data,
                amplitude_indices=amplitude_indices,
                main_predictions=fold_main[fold_index - 1],
                external_frame=external_frame,
                nag_root=Path(args.nag_root).resolve(),
                args=args,
            )
            seed_folds.append(prediction)
            seed_details.append(detail)
        seed_stack = np.stack(seed_folds)
        final = hard_vote_with_probability_tiebreak(seed_stack, support)
        all_seed_predictions.append(final)
        all_seed_folds.append(seed_stack)
        diagnostics.append(
            {
                "seed": int(seed),
                "final_change_fraction": float(np.mean(final != frozen_main)),
                "folds": seed_details,
            }
        )
        print(
            f"seed {seed}: final change from frozen main={np.mean(final != frozen_main):.4%}",
            flush=True,
        )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        seeds=np.asarray(args.seeds, dtype=np.int64),
        row_indices=np.arange(len(external_frame), dtype=np.int64),
        main_predictions=frozen_main,
        seed_predictions=np.stack(all_seed_predictions),
        seed_fold_predictions=np.stack(all_seed_folds),
    )
    digest = sha256_file(output)
    save_json(
        {
            "method": "post-reveal prespecified label-free NAG seed stability",
            "labels_used_for_prediction": False,
            "seeds": args.seeds,
            "diagnostics": diagnostics,
            "prediction_sha256": digest,
        },
        output.with_suffix(".json"),
    )
    output.with_suffix(".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="ascii"
    )
    print(f"saved {output}; sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
