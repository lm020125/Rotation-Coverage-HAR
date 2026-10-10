from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from copy import deepcopy
from pathlib import Path

from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcbd_har.config import load_config  # noqa: E402
from rcbd_har.data import WindowManifestDataset, estimate_global_normalization, load_manifest  # noqa: E402
from rcbd_har.engine import evaluate, fit  # noqa: E402
from rcbd_har.models import build_model  # noqa: E402
from rcbd_har.utils import resolve_device, save_json, seed_everything  # noqa: E402


DEFAULT_CONFIGS = [
    "configs/motionminers_clean.yaml",
    "configs/motionminers_uniform180_pc.yaml",
    "configs/motionminers_fmc_pc.yaml",
]
DEFAULT_SEEDS = [256, 3407, 114514]


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_kwargs(config: dict, stats: dict) -> dict:
    return {
        "normalization": config["data"]["normalization"],
        "use_audio": False,
        "limb_mode": "bilateral",
        "normalization_stats": stats,
        "missing_strategy": config["data"].get("missing_strategy", "error"),
    }


def _loader(dataset: WindowManifestDataset, config: dict, device, shuffle: bool) -> DataLoader:
    workers = int(config["training"].get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def _resolved_config(path: Path, seed: int) -> dict:
    config = load_config(path)
    config["seed"] = seed
    return config


def _selection(config_path: Path, seed: int, output: Path, device_name: str | None) -> None:
    config = _resolved_config(config_path, seed)
    frame = load_manifest(config["data"]["manifest"])
    participants = sorted(frame.participant.unique().tolist())
    if participants != ["S01", "S02", "S03"]:
        raise ValueError(f"Source-only selection requires S01-S03, got {participants}")
    device = resolve_device(device_name)
    fold_epochs: list[int] = []
    fold_scores: list[float] = []
    for fold, val_participant in enumerate(participants, start=1):
        fold_config = deepcopy(config)
        train_frame = frame.loc[frame.participant != val_participant].reset_index(drop=True)
        val_frame = frame.loc[frame.participant == val_participant].reset_index(drop=True)
        stats = estimate_global_normalization(train_frame, "bilateral", mode="train_rms_shared")
        fold_config["data"]["normalization_stats"] = stats
        kwargs = _dataset_kwargs(fold_config, stats)
        train_dataset = WindowManifestDataset(
            train_frame, swap_probability=0.0, **kwargs
        )
        val_dataset = WindowManifestDataset(val_frame, **kwargs)
        seed_everything(seed + fold * 1000)
        model = build_model(fold_config).to(device)
        fold_dir = output / config_path.stem / f"seed{seed}" / "selection" / f"val_{val_participant}"
        _, history = fit(
            model, _loader(train_dataset, fold_config, device, True),
            _loader(val_dataset, fold_config, device, False), fold_config, device, fold_dir,
        )
        best = max(history, key=lambda item: item.get("val_selection_score", item["val_macro_f1"]))
        fold_epochs.append(int(best["epoch"]))
        fold_scores.append(float(best.get("val_selection_score", best["val_macro_f1"])))
    selected_epoch = int(statistics.median(fold_epochs))
    save_json(
        {
            "method": config_path.stem,
            "seed": seed,
            "selection_participants": participants,
            "fold_best_epochs": fold_epochs,
            "fold_best_macro_f1": fold_scores,
            "selected_epoch": selected_epoch,
            "target_manifests_accessed": False,
        },
        output / config_path.stem / f"seed{seed}" / "selected_epoch.json",
    )


def _evaluate(config_path: Path, seed: int, output: Path, device_name: str | None) -> dict:
    config = _resolved_config(config_path, seed)
    source_path = Path(config["data"]["manifest"])
    data_dir = source_path.parent
    source = load_manifest(source_path)
    correct = load_manifest(data_dir / "manifest_target_correct.csv")
    misplaced = load_manifest(data_dir / "manifest_target_misplaced.csv")
    if set(source.participant) & set(correct.participant):
        raise ValueError("Source and target participants overlap")
    if correct.pair_id.tolist() != misplaced.pair_id.tolist():
        raise ValueError("Target correct and misplaced windows are not paired")
    selected_path = output / config_path.stem / f"seed{seed}" / "selected_epoch.json"
    selection = json.loads(selected_path.read_text(encoding="utf-8"))
    selected_epoch = int(selection["selected_epoch"])
    stats = estimate_global_normalization(source, "bilateral", mode="train_rms_shared")
    config["data"]["normalization_stats"] = stats
    kwargs = _dataset_kwargs(config, stats)
    source_dataset = WindowManifestDataset(source, swap_probability=0.0, **kwargs)
    correct_dataset = WindowManifestDataset(correct, **kwargs)
    misplaced_dataset = WindowManifestDataset(misplaced, **kwargs)
    device = resolve_device(device_name)
    seed_everything(seed + 10_000)
    model = build_model(config).to(device)
    run_dir = output / config_path.stem / f"seed{seed}" / "final"
    model, history = fit(
        model, _loader(source_dataset, config, device, True), None, config, device,
        run_dir, fixed_epochs=selected_epoch,
    )
    correct_metrics = evaluate(
        model, _loader(correct_dataset, config, device, False), device, 2,
        swap_stress=False, smoothing={"enabled": False},
    )
    misplaced_metrics = evaluate(
        model, _loader(misplaced_dataset, config, device, False), device, 2,
        swap_stress=False, smoothing={"enabled": False},
    )
    result = {
        "method": config_path.stem,
        "seed": seed,
        "selected_epoch": selected_epoch,
        "source_participants": sorted(source.participant.unique().tolist()),
        "target_participants": sorted(correct.participant.unique().tolist()),
        "correct": correct_metrics,
        "misplaced": misplaced_metrics,
        "macro_f1_drop": correct_metrics["macro_f1"] - misplaced_metrics["macro_f1"],
        "macro_f1_retention": misplaced_metrics["macro_f1"] / max(correct_metrics["macro_f1"], 1e-12),
    }
    save_json(history, run_dir / "history.json")
    save_json(stats, run_dir / "normalization.json")
    save_json(result, run_dir / "external_metrics.json")
    return result


def _lock_payload(configs: list[Path]) -> dict:
    protected = [
        Path(__file__).resolve(),
        ROOT / "scripts" / "prepare_motionminers_mp1.py",
        ROOT / "src" / "rcbd_har" / "engine.py",
        ROOT / "src" / "rcbd_har" / "placement.py",
        *[path.resolve() for path in configs],
    ]
    for config_path in configs:
        profile = load_config(config_path)["training"].get("quality_augmentation_profile_path")
        if profile and Path(profile).exists():
            protected.append(Path(profile).resolve())
    return {"sha256": {str(path): _hash(path) for path in protected}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict source-only-selected MP1 external validation")
    parser.add_argument("--stage", choices=("select", "evaluate", "all"), required=True)
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    configs = [Path(value).resolve() for value in args.configs]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "protocol_lock.json"
    if args.stage in {"select", "all"}:
        save_json(_lock_payload(configs), lock_path)
        for config in configs:
            for seed in args.seeds:
                _selection(config, seed, output, args.device)
    if args.stage in {"evaluate", "all"}:
        expected = json.loads(lock_path.read_text(encoding="utf-8"))
        observed = _lock_payload(configs)
        if observed != expected:
            raise RuntimeError("Protocol/code/config hashes changed after source-only selection")
        results = []
        for config in configs:
            for seed in args.seeds:
                results.append(_evaluate(config, seed, output, args.device))
        save_json(results, output / "all_external_metrics.json")


if __name__ == "__main__":
    main()
