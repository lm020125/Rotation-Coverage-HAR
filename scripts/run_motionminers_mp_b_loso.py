from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcbd_har.config import load_config  # noqa: E402
from rcbd_har.data import (  # noqa: E402
    WindowManifestDataset,
    estimate_global_normalization,
    load_manifest,
)
from rcbd_har.engine import evaluate, fit  # noqa: E402
from rcbd_har.models import build_model  # noqa: E402
from rcbd_har.placement import estimate_placement_profile  # noqa: E402
from rcbd_har.utils import resolve_device, save_json, seed_everything  # noqa: E402


PARTICIPANTS = ("S04", "S05", "S06", "S07")
VALIDATION_PARTICIPANT = {
    "S04": "S05",
    "S05": "S06",
    "S06": "S07",
    "S07": "S04",
}
METHODS = ("clean", "uniform180", "uniform180_pc", "empirical", "fmc_pc")
SEEDS = (3407, 256, 114514)
SUBSETS = ("full", "arm_affected", "torso_control")


def _dataset_kwargs(config: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "normalization": config["data"]["normalization"],
        "use_audio": False,
        "limb_mode": "bilateral",
        "normalization_stats": stats,
        "missing_strategy": config["data"].get("missing_strategy", "error"),
    }


def _loader(
    dataset: WindowManifestDataset,
    config: dict[str, Any],
    device: Any,
    shuffle: bool,
) -> DataLoader:
    workers = int(config["training"].get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def _condition_config(
    base: dict[str, Any], method: str, seed: int, profile_path: Path | None
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")
    config = deepcopy(base)
    config["seed"] = int(seed)
    training = config["training"]
    loss = config["loss"]
    training.pop("quality_augmentation_profile_path", None)
    training.pop("quality_augmentation_max_degrees", None)
    training["quality_augmentation_probability"] = 0.0 if method == "clean" else 0.5
    loss["perturbation_prediction_consistency"] = (
        0.1 if method in {"uniform180_pc", "fmc_pc"} else 0.0
    )
    if method in {"uniform180", "uniform180_pc"}:
        training["quality_augmentation_max_degrees"] = 180.0
    if method in {"empirical", "fmc_pc"}:
        if profile_path is None:
            raise ValueError(f"{method} requires a fold-specific profile")
        training["quality_augmentation_profile_path"] = str(profile_path.resolve())
    return config


def _unlabeled_record_view(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["participant", "recording_id", "left_path", "right_path", "num_samples"]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"Profile view lacks {sorted(missing)}")
    return frame[required].drop_duplicates().sort_values(["participant", "recording_id"])


def build_profile_bank(
    correct: pd.DataFrame,
    misplaced: pd.DataFrame,
    participants: list[str],
    output: Path,
) -> dict[str, Any]:
    correct_view = _unlabeled_record_view(
        correct.loc[correct.participant.isin(participants)].reset_index(drop=True)
    )
    misplaced_view = _unlabeled_record_view(
        misplaced.loc[misplaced.participant.isin(participants)].reset_index(drop=True)
    )
    paired = correct_view.merge(
        misplaced_view,
        on=["participant", "recording_id", "num_samples"],
        suffixes=("_correct", "_misplaced"),
        validate="one_to_one",
    )
    payload: dict[str, Any] = {
        "protocol": {
            "name": "MotionMiners MP_B training-participant paired profile bank",
            "labels_read": False,
            "activity_labels_used": False,
            "misplacement_annotations_used": False,
            "participants": participants,
            "profile_unit": "one record per participant and arm",
            "sampling": "participant first, then record",
        },
        "limbs": {"left_wrist": [], "right_wrist": []},
        "excluded": [],
    }
    for row in paired.itertuples(index=False):
        for side, limb in (("left", "left_wrist"), ("right", "right_wrist")):
            try:
                profile = estimate_placement_profile(
                    np.load(
                        getattr(row, f"{side}_path_correct"),
                        mmap_mode="r",
                        allow_pickle=False,
                    ),
                    np.load(
                        getattr(row, f"{side}_path_misplaced"),
                        mmap_mode="r",
                        allow_pickle=False,
                    ),
                )
                profile.update(
                    participant=str(row.participant),
                    recording_id=str(row.recording_id),
                )
                payload["limbs"][limb].append(profile)
            except ValueError as error:
                payload["excluded"].append(
                    {
                        "participant": str(row.participant),
                        "recording_id": str(row.recording_id),
                        "limb": limb,
                        "reason": str(error),
                    }
                )
    for limb, profiles in payload["limbs"].items():
        represented = {str(profile["participant"]) for profile in profiles}
        if represented != set(participants):
            raise RuntimeError(
                f"Profile bank {limb} does not represent every training participant: {represented}"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _filter_subset(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "full":
        selected = frame
    elif subset == "arm_affected":
        selected = frame.loc[frame["arm_input_affected"].astype(str).str.lower().eq("true")]
    elif subset == "torso_control":
        selected = frame.loc[frame["misplacement_group"].eq("torso_only")]
    else:
        raise ValueError(f"Unknown evaluation subset {subset}")
    selected = selected.reset_index(drop=True)
    if selected.empty or set(selected.label.unique()) != {0, 1}:
        raise ValueError(f"Subset {subset} must contain both HAR classes")
    return selected


def _evaluate_subsets(
    model: Any,
    correct: pd.DataFrame,
    misplaced: pd.DataFrame,
    config: dict[str, Any],
    stats: dict[str, Any],
    device: Any,
) -> dict[str, Any]:
    kwargs = _dataset_kwargs(config, stats)
    output: dict[str, Any] = {}
    for subset in SUBSETS:
        correct_subset = _filter_subset(correct, subset)
        misplaced_subset = _filter_subset(misplaced, subset)
        if correct_subset.pair_id.tolist() != misplaced_subset.pair_id.tolist():
            raise RuntimeError(f"Paired windows diverged in subset {subset}")
        correct_metrics = evaluate(
            model,
            _loader(WindowManifestDataset(correct_subset, **kwargs), config, device, False),
            device,
            2,
            swap_stress=False,
            smoothing={"enabled": False},
        )
        misplaced_metrics = evaluate(
            model,
            _loader(WindowManifestDataset(misplaced_subset, **kwargs), config, device, False),
            device,
            2,
            swap_stress=False,
            smoothing={"enabled": False},
        )
        output[subset] = {
            "windows": len(correct_subset),
            "correct": correct_metrics,
            "misplaced": misplaced_metrics,
            "macro_f1_drop": float(correct_metrics["macro_f1"] - misplaced_metrics["macro_f1"]),
        }
    return output


def run_fold(
    base: dict[str, Any],
    correct: pd.DataFrame,
    misplaced: pd.DataFrame,
    method: str,
    seed: int,
    test_participant: str,
    output_root: Path,
    device_name: str | None,
) -> dict[str, Any]:
    fold_dir = output_root / method / f"seed{seed}" / f"test_{test_participant}"
    metrics_path = fold_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    val_participant = VALIDATION_PARTICIPANT[test_participant]
    outer_train = [name for name in PARTICIPANTS if name != test_participant]
    selection_train = [name for name in outer_train if name != val_participant]
    empirical = method in {"empirical", "fmc_pc"}
    selection_profile = None
    if empirical:
        selection_profile = fold_dir / "selection_profile.json"
        build_profile_bank(correct, misplaced, selection_train, selection_profile)
    selection_config = _condition_config(base, method, seed, selection_profile)
    selection_frame = correct.loc[
        correct.participant.isin(selection_train + [val_participant])
    ].reset_index(drop=True)
    train_frame = selection_frame.loc[
        selection_frame.participant.isin(selection_train)
    ].reset_index(drop=True)
    val_frame = selection_frame.loc[selection_frame.participant.eq(val_participant)].reset_index(
        drop=True
    )
    stats = estimate_global_normalization(train_frame, "bilateral", mode="train_rms_shared")
    selection_config["data"]["normalization_stats"] = stats
    kwargs = _dataset_kwargs(selection_config, stats)
    device = resolve_device(device_name)
    seed_everything(seed + PARTICIPANTS.index(test_participant) * 1000)
    model = build_model(selection_config).to(device)
    selection_dir = fold_dir / "selection"
    _, selection_history = fit(
        model,
        _loader(
            WindowManifestDataset(train_frame, swap_probability=0.0, **kwargs),
            selection_config,
            device,
            True,
        ),
        _loader(WindowManifestDataset(val_frame, **kwargs), selection_config, device, False),
        selection_config,
        device,
        selection_dir,
    )
    best = max(
        selection_history,
        key=lambda item: item.get("val_selection_score", item["val_macro_f1"]),
    )
    selected_epoch = int(best["epoch"])

    refit_profile = None
    if empirical:
        refit_profile = fold_dir / "refit_profile.json"
        build_profile_bank(correct, misplaced, outer_train, refit_profile)
    refit_config = _condition_config(base, method, seed, refit_profile)
    refit_frame = correct.loc[correct.participant.isin(outer_train)].reset_index(drop=True)
    refit_stats = estimate_global_normalization(refit_frame, "bilateral", mode="train_rms_shared")
    refit_config["data"]["normalization_stats"] = refit_stats
    refit_kwargs = _dataset_kwargs(refit_config, refit_stats)
    seed_everything(seed + PARTICIPANTS.index(test_participant) * 1000 + 10_000)
    model = build_model(refit_config).to(device)
    model, refit_history = fit(
        model,
        _loader(
            WindowManifestDataset(refit_frame, swap_probability=0.0, **refit_kwargs),
            refit_config,
            device,
            True,
        ),
        None,
        refit_config,
        device,
        fold_dir / "refit",
        fixed_epochs=selected_epoch,
    )
    test_correct = correct.loc[correct.participant.eq(test_participant)].reset_index(drop=True)
    test_misplaced = misplaced.loc[misplaced.participant.eq(test_participant)].reset_index(
        drop=True
    )
    subset_metrics = _evaluate_subsets(
        model, test_correct, test_misplaced, refit_config, refit_stats, device
    )
    result = {
        "method": method,
        "seed": seed,
        "test_participant": test_participant,
        "validation_participant": val_participant,
        "selection_train_participants": selection_train,
        "refit_train_participants": outer_train,
        "selected_epoch": selected_epoch,
        "selection_best_macro_f1": float(best["val_macro_f1"]),
        "subsets": subset_metrics,
        "protocol": {
            "test_data_used_for_profiles": False,
            "test_data_used_for_selection": False,
            "refit_full_outer_train": True,
            "smoothing": False,
            "profile_participant_balanced": empirical,
        },
    }
    save_json(selection_history, selection_dir / "history.json")
    save_json(refit_history, fold_dir / "refit" / "history.json")
    save_json(refit_stats, fold_dir / "refit" / "normalization.json")
    save_json(result, metrics_path)
    return result


def summarize(output_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in output_root.glob("*/seed*/test_*/metrics.json"):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        raise ValueError("No completed fold metrics are available")
    seed_summary: dict[str, Any] = defaultdict(dict)
    final_summary: dict[str, Any] = {}
    for method in METHODS:
        method_records = [record for record in records if record["method"] == method]
        if not method_records:
            continue
        for seed in SEEDS:
            selected = [record for record in method_records if record["seed"] == seed]
            if len(selected) != len(PARTICIPANTS):
                continue
            seed_summary[method][str(seed)] = {}
            for subset in SUBSETS:
                seed_summary[method][str(seed)][subset] = {}
                for condition in ("correct", "misplaced"):
                    values = [
                        float(record["subsets"][subset][condition]["macro_f1"])
                        for record in selected
                    ]
                    seed_summary[method][str(seed)][subset][condition] = {
                        "participant_equal_macro_f1_mean": statistics.mean(values),
                        "participant_macro_f1_std": statistics.stdev(values),
                        "per_participant": {
                            record["test_participant"]: float(
                                record["subsets"][subset][condition]["macro_f1"]
                            )
                            for record in selected
                        },
                    }
        complete_seeds = [int(seed) for seed in seed_summary[method] if int(seed) in SEEDS]
        if complete_seeds:
            final_summary[method] = {}
            for subset in SUBSETS:
                final_summary[method][subset] = {}
                for condition in ("correct", "misplaced"):
                    values = [
                        seed_summary[method][str(seed)][subset][condition][
                            "participant_equal_macro_f1_mean"
                        ]
                        for seed in complete_seeds
                    ]
                    final_summary[method][subset][condition] = {
                        "seed_mean": statistics.mean(values),
                        "seed_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "seeds": complete_seeds,
                    }
    payload = {
        "primary_metric": "participant-equal mean of held-out-participant Macro-F1",
        "seed_summary": seed_summary,
        "final_summary": final_summary,
        "completed_fold_runs": len(records),
    }
    save_json(payload, output_root / "summary.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="MotionMiners MP_B paired LOSO experiment")
    parser.add_argument("--stage", choices=("run", "summarize", "all"), default="all")
    parser.add_argument("--config", default="configs/motionminers_mp_b_loso.yaml")
    parser.add_argument("--correct-manifest")
    parser.add_argument("--misplaced-manifest")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--folds", nargs="+", choices=PARTICIPANTS, default=list(PARTICIPANTS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.stage in {"run", "all"}:
        base = load_config(args.config)
        correct_path = Path(args.correct_manifest or base["data"]["manifest"]).resolve()
        misplaced_path = Path(
            args.misplaced_manifest or correct_path.with_name("manifest_misplaced.csv")
        ).resolve()
        correct = load_manifest(correct_path)
        misplaced = load_manifest(misplaced_path)
        if tuple(sorted(correct.participant.unique().tolist())) != PARTICIPANTS:
            raise ValueError("The MP_B LOSO manifest must contain exactly S04-S07")
        if correct.pair_id.tolist() != misplaced.pair_id.tolist():
            raise ValueError("Correct and misplaced manifests are not exactly paired")
        for seed in args.seeds:
            for method in args.methods:
                for test_participant in args.folds:
                    run_fold(
                        base,
                        correct,
                        misplaced,
                        method,
                        seed,
                        test_participant,
                        output_root,
                        args.device,
                    )
    if args.stage in {"summarize", "all"}:
        print(json.dumps(summarize(output_root), indent=2))


if __name__ == "__main__":
    main()
