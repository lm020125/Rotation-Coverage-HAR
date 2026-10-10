from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rcbd_har.config import load_config  # noqa: E402
from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.experiment import run_group_fold  # noqa: E402
from rcbd_har.utils import participant_sort_key, save_json  # noqa: E402
from scripts.run_grouped_cv import (  # noqa: E402
    aggregate_fold_metrics,
    load_completed_fold,
    load_folds,
)


METHODS = (
    "clean",
    "lr_swap",
    "uniform180",
    "uniform180_pc",
    "fmc_no_pc",
    "fmc_pc",
)
DEFAULT_SEEDS = (256, 3407, 114514)
EXTERNAL_PARTICIPANTS = {f"sbj_{index}" for index in range(18, 22)}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_profile(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol", {}).get("labels_read") is not False:
        raise ValueError("FMC profile must explicitly attest labels_read=false")
    limbs = payload.get("limbs", {})
    required = {"left_wrist", "right_wrist", "left_ankle", "right_ankle"}
    if not required.issubset(limbs) or any(not limbs[key] for key in required):
        raise ValueError("FMC profile must contain non-empty profiles for all four limbs")
    profile_people = {str(item.get("participant", "")) for key in required for item in limbs[key]}
    if profile_people & EXTERNAL_PARTICIPANTS:
        raise ValueError("External WEAR participants must not enter the FMC profile")
    return payload


def condition_config(
    base: dict[str, Any], method: str, seed: int, profile_path: Path
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"Unknown WEAR condition: {method}")
    config = copy.deepcopy(base)
    config["seed"] = int(seed)
    training = config["training"]
    loss = config["loss"]
    for key in (
        "quality_augmentation_profile_path",
        "quality_augmentation_max_degrees",
    ):
        training.pop(key, None)
    training.update(
        {
            "swap_augmentation_probability": 0.0,
            "orientation_augmentation_probability": 0.0,
            "quality_augmentation_probability": 0.0,
            "quality_augmentation_classification_mix": 0.5,
            "quality_augmentation_profile_sampling": "gravity_matched_random",
            "quality_augmentation_profile_rotation": True,
            "quality_augmentation_profile_gain": True,
            "quality_augmentation_profile_noise": True,
            "quality_augmentation_profile_participant_balanced": True,
            "quality_augmentation_exclude_source_participant": False,
        }
    )
    loss["perturbation_prediction_consistency"] = 0.0
    if method == "lr_swap":
        training["swap_augmentation_probability"] = 0.5
    elif method in {"uniform180", "uniform180_pc"}:
        training["quality_augmentation_probability"] = 0.5
        training["quality_augmentation_max_degrees"] = 180.0
    elif method in {"fmc_no_pc", "fmc_pc"}:
        training["quality_augmentation_probability"] = 0.5
        training["quality_augmentation_profile_path"] = str(profile_path.resolve())
    if method in {"uniform180_pc", "fmc_pc"}:
        loss["perturbation_prediction_consistency"] = 0.1
    return config


def _validation_people(
    participants: list[str], test_people: list[str], fold_index: int, count: int
) -> list[str]:
    remaining = [person for person in participants if person not in test_people]
    offset = (fold_index * count) % len(remaining)
    rotated = remaining[offset:] + remaining[:offset]
    return rotated[:count]


def _protocol_payload(
    config_path: Path,
    folds_path: Path,
    profile_path: Path,
    folds: list[list[str]],
) -> dict[str, Any]:
    return {
        "name": "WEAR four-sensor participant-disjoint FMC-PC validation",
        "config_sha256": _sha256(config_path),
        "folds_sha256": _sha256(folds_path),
        "profile_sha256": _sha256(profile_path),
        "test_folds": folds,
        "external_participants_accessed": False,
        "input": "four 3-axis accelerometers at 50 Hz",
        "classes": 19,
        "primary_metric": "sample-wise macro-F1 with common smoothing",
        "secondary_metric": "unsmoothed center-window macro-F1",
        "augmentation_probability": 0.5,
        "prediction_consistency_weight": 0.1,
        "seeds": list(DEFAULT_SEEDS),
        "methods": list(METHODS),
    }


def _write_or_verify_lock(output: Path, payload: dict[str, Any]) -> None:
    path = output / "protocol_lock.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != payload:
            raise ValueError("Existing WEAR output has a different frozen protocol")
    else:
        save_json(payload, path)


def summarize(output: Path, methods: list[str], seeds: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {"methods": {}, "completed_fold_runs": 0}
    for method in methods:
        method_payload: dict[str, Any] = {"seeds": {}}
        seed_values: list[float] = []
        for seed in seeds:
            paths = sorted((output / method / f"seed_{seed}").glob("fold_*/metrics.json"))
            result["completed_fold_runs"] += len(paths)
            if len(paths) != 3:
                continue
            fold_metrics = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
            aggregate = aggregate_fold_metrics(fold_metrics)
            samplewise = float(aggregate.get("samplewise", aggregate)["macro_f1"])
            center = float(aggregate["macro_f1"])
            participant_scores = {
                participant: float(metrics["macro_f1"])
                for participant, metrics in aggregate.get("samplewise", {})
                .get("per_participant", {})
                .items()
            }
            method_payload["seeds"][str(seed)] = {
                "samplewise_macro_f1": samplewise,
                "center_macro_f1": center,
                "fold_macro_f1": [float(item["macro_f1"]) for item in fold_metrics],
                "per_participant_macro_f1": participant_scores,
            }
            seed_values.append(samplewise)
        if seed_values:
            method_payload["samplewise_macro_f1_mean"] = statistics.mean(seed_values)
            method_payload["samplewise_macro_f1_std"] = (
                statistics.stdev(seed_values) if len(seed_values) > 1 else 0.0
            )
        result["methods"][method] = method_payload
    save_json(result, output / "summary.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict WEAR FMC-PC grouped CV")
    parser.add_argument("--config", default="configs/wear_rcbd.yaml")
    parser.add_argument("--folds", default="configs/wear_paper_folds.json")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--fold-indices", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    folds_path = Path(args.folds).resolve()
    profile_path = Path(args.profile).resolve()
    validate_profile(profile_path)
    base = load_config(config_path)
    frame = load_manifest(base["data"]["manifest"])
    participants = sorted(frame.participant.astype(str).unique().tolist(), key=participant_sort_key)
    if set(participants) != {f"sbj_{index}" for index in range(18)}:
        raise ValueError("Development manifest must contain exactly WEAR sbj_0 through sbj_17")
    if set(participants) & EXTERNAL_PARTICIPANTS:
        raise ValueError("External WEAR participants entered the development manifest")
    folds = load_folds(str(folds_path), participants, 3)
    selected = [index - 1 for index in args.fold_indices]
    if not selected or any(index not in range(3) for index in selected):
        raise ValueError("WEAR fold indices must be 1, 2, or 3")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = _protocol_payload(config_path, folds_path, profile_path, folds)
    _write_or_verify_lock(output, lock)

    if not args.summarize_only:
        val_count = int(base.get("cross_validation", {}).get("val_participants_per_fold", 2))
        for seed in args.seeds:
            for method in args.methods:
                for fold_index in selected:
                    test_people = folds[fold_index]
                    val_people = _validation_people(
                        participants, test_people, fold_index, val_count
                    )
                    fold_output = output / method / f"seed_{seed}" / f"fold_{fold_index + 1}"
                    existing = load_completed_fold(fold_output, test_people, val_people)
                    if existing is not None:
                        print(f"Reusing {method} seed={seed} fold={fold_index + 1}")
                        continue
                    config = condition_config(base, method, seed, profile_path)
                    run_group_fold(
                        config,
                        test_people,
                        val_people,
                        fold_output,
                        args.device,
                        refit_full_train=True,
                    )
                    summarize(output, list(METHODS), list(DEFAULT_SEEDS))
    print(json.dumps(summarize(output, list(METHODS), list(DEFAULT_SEEDS)), indent=2))


if __name__ == "__main__":
    main()
