from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcbd_har.config import load_config, with_overrides  # noqa: E402
from rcbd_har.data import (  # noqa: E402
    WindowManifestDataset,
    estimate_global_normalization,
    load_manifest,
)
from rcbd_har.engine import _axis_angle_rotation_matrices, fit, forward_batch  # noqa: E402
from rcbd_har.models import build_model  # noqa: E402
from rcbd_har.reliability import (  # noqa: E402
    PAIR_NAMES,
    relative_discriminative_retention,
)
from rcbd_har.utils import move_batch, resolve_device, save_json, seed_everything  # noqa: E402


SOURCE_PARTICIPANTS = tuple(f"subject{value}" for value in range(1, 11))
FINAL_TEST_PARTICIPANTS = tuple(f"subject{value}" for value in range(13, 18))
CROSSFIT_FOLDS = (
    ("subject1", "subject6"),
    ("subject2", "subject7"),
    ("subject3", "subject8"),
    ("subject4", "subject9"),
    ("subject5", "subject10"),
)
PAIR_KEYS = {
    "arm": ("left_wrist", "right_wrist"),
    "leg": ("left_ankle", "right_ankle"),
}
DEFAULT_EDGES = (0.0, 15.0, 30.0, 60.0, 120.0, 180.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def controlled_pair_rotation(
    batch: dict[str, Any], pair: str, degrees: float, seed: int
) -> dict[str, Any]:
    """Rotate exactly one side of one bilateral pair at a fixed severity."""
    if pair not in PAIR_KEYS:
        raise ValueError(f"Unknown pair: {pair}")
    left_key, right_key = PAIR_KEYS[pair]
    left, right = batch[left_key], batch[right_key]
    batch_size = left.shape[0]
    devices = [left.device] if left.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if left.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        severity = torch.ones(batch_size, device=left.device, dtype=left.dtype)
        rotation = _axis_angle_rotation_matrices(severity, degrees, left.dtype)
        choose_left = torch.rand(batch_size, device=left.device) < 0.5
    transformed_left = torch.einsum("bij,btj->bti", rotation, left)
    transformed_right = torch.einsum("bij,btj->bti", rotation, right)
    augmented = dict(batch)
    augmented[left_key] = torch.where(choose_left.view(-1, 1, 1), transformed_left, left)
    augmented[right_key] = torch.where((~choose_left).view(-1, 1, 1), transformed_right, right)
    return augmented


def dataset_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "normalization": config["data"].get("normalization", "scale_only"),
        "use_audio": bool(config["data"].get("use_audio", False)),
        "limb_mode": config["data"].get("limb_mode", "bilateral"),
        "normalization_stats": config["data"].get("normalization_stats"),
        "normalization_scale": float(config["data"].get("normalization_scale", 8.0)),
        "missing_strategy": config["data"].get("missing_strategy", "error"),
    }


@torch.no_grad()
def accumulate_fold_reliability(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
    centers: np.ndarray,
    repeats: int,
    temperature: float,
    seed: int,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((num_classes, len(PAIR_NAMES), len(centers)), dtype=np.float64)
    counts = np.zeros_like(sums, dtype=np.int64)
    model.eval()
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(raw_batch, device)
        labels = batch["label"]
        label_values = labels.detach().cpu().numpy()
        class_counts = np.bincount(label_values, minlength=num_classes)
        clean_logits = forward_batch(model, batch)["logits"]
        for pair_index, pair in enumerate(PAIR_NAMES):
            for bin_index, degrees in enumerate(centers):
                retention = torch.zeros(len(labels), device=device)
                for repeat in range(repeats):
                    augmented = controlled_pair_rotation(
                        batch,
                        pair,
                        float(degrees),
                        seed
                        + 100_000 * batch_index
                        + 10_000 * pair_index
                        + 100 * bin_index
                        + repeat,
                    )
                    augmented_logits = forward_batch(model, augmented)["logits"]
                    retention += relative_discriminative_retention(
                        clean_logits, augmented_logits, labels, temperature
                    )
                retention = retention / float(repeats)
                sums[:, pair_index, bin_index] += np.bincount(
                    label_values,
                    weights=retention.detach().cpu().numpy(),
                    minlength=num_classes,
                )
                counts[:, pair_index, bin_index] += class_counts
    return sums, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build source-only cross-fitted CR-FMC-PC reliability statistics."
    )
    parser.add_argument("--config", default="configs/realdisp_rcbd.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fixed-epochs", type=int, default=13)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--shrinkage", type=float, default=20.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--max-holdout-batches", type=int)
    args = parser.parse_args()
    if args.fixed_epochs < 1 or args.repeats < 1:
        raise ValueError("fixed-epochs and repeats must be positive")
    if args.max_folds is not None and args.max_folds < 1:
        raise ValueError("max-folds must be positive")
    if args.max_holdout_batches is not None and args.max_holdout_batches < 1:
        raise ValueError("max-holdout-batches must be positive")
    if args.temperature <= 0 or args.shrinkage < 0:
        raise ValueError("temperature must be positive and shrinkage non-negative")

    config_path = (ROOT / args.config).resolve()
    base_config = with_overrides(
        load_config(config_path),
        seed=args.seed,
        **{
            "training.num_workers": args.num_workers,
            "training.batch_size": args.batch_size,
        },
    )
    device = resolve_device(args.device)
    frame = load_manifest(base_config["data"]["manifest"])
    participant_series = frame["participant"].astype(str)
    available = set(participant_series.unique())
    if missing := set(SOURCE_PARTICIPANTS) - available:
        raise ValueError(f"Source participants are absent from the manifest: {sorted(missing)}")
    if set(FINAL_TEST_PARTICIPANTS) & set(SOURCE_PARTICIPANTS):
        raise RuntimeError("Final-test participants overlap the source participants")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    edges = np.asarray(DEFAULT_EDGES, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    num_classes = int(base_config["data"]["num_classes"])
    total_sums = np.zeros((num_classes, len(PAIR_NAMES), len(centers)), dtype=np.float64)
    total_counts = np.zeros_like(total_sums, dtype=np.int64)
    fold_protocols: list[dict[str, Any]] = []

    selected_folds = CROSSFIT_FOLDS if args.max_folds is None else CROSSFIT_FOLDS[: args.max_folds]
    if not selected_folds:
        raise ValueError("max-folds must select at least one fold")
    for fold_index, holdout in enumerate(selected_folds):
        fold_seed = args.seed + fold_index
        seed_everything(fold_seed)
        train_participants = tuple(
            participant for participant in SOURCE_PARTICIPANTS if participant not in holdout
        )
        train_indices = frame.index[participant_series.isin(train_participants)].tolist()
        holdout_indices = frame.index[participant_series.isin(holdout)].tolist()
        config = load_config(config_path)
        config = with_overrides(
            config,
            seed=fold_seed,
            **{
                "training.num_workers": args.num_workers,
                "training.batch_size": args.batch_size,
            },
        )
        normalization = config["data"].get("normalization", "scale_only")
        if normalization in {"train_global", "train_rms_shared"}:
            config["data"]["normalization_stats"] = estimate_global_normalization(
                frame.iloc[train_indices],
                config["data"].get("limb_mode", "bilateral"),
                mode=normalization,
            )
        kwargs = dataset_kwargs(config)
        train_dataset = WindowManifestDataset(frame, train_indices, **kwargs)
        holdout_dataset = WindowManifestDataset(frame, holdout_indices, **kwargs)
        loader_args = {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": device.type == "cuda",
            "persistent_workers": args.num_workers > 0,
        }
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
        holdout_loader = DataLoader(holdout_dataset, shuffle=False, **loader_args)
        fold_root = output / f"fold{fold_index + 1}"
        checkpoint = fold_root / "teacher.pt"
        model = build_model(config).to(device)
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
        else:
            model, history = fit(
                model,
                train_loader,
                None,
                config,
                device,
                fold_root,
                fixed_epochs=args.fixed_epochs,
                checkpoint_name="teacher.pt",
            )
            save_json(history, fold_root / "history.json")
        sums, counts = accumulate_fold_reliability(
            model,
            holdout_loader,
            device=device,
            num_classes=num_classes,
            centers=centers,
            repeats=args.repeats,
            temperature=args.temperature,
            seed=fold_seed + 1_000_000,
            max_batches=args.max_holdout_batches,
        )
        total_sums += sums
        total_counts += counts
        fold_record = {
            "fold": fold_index + 1,
            "seed": fold_seed,
            "train_participants": list(train_participants),
            "holdout_participants": list(holdout),
            "fixed_epochs": args.fixed_epochs,
            "train_windows": len(train_dataset),
            "holdout_windows": len(holdout_dataset),
            "holdout_never_used_for_training_or_checkpoint_selection": True,
        }
        fold_protocols.append(fold_record)
        save_json(fold_record, fold_root / "protocol.json")
        save_json(
            {"sums": sums.tolist(), "counts": counts.tolist()},
            fold_root / "retention_sufficient_statistics.json",
        )

    global_sums = total_sums.sum(axis=0)
    global_counts = total_counts.sum(axis=0)
    global_values = np.divide(
        global_sums,
        np.maximum(global_counts, 1),
        out=np.ones_like(global_sums),
        where=global_counts > 0,
    )
    raw_class = np.divide(
        total_sums,
        np.maximum(total_counts, 1),
        out=np.broadcast_to(global_values, total_sums.shape).copy(),
        where=total_counts > 0,
    )
    class_values = (
        total_counts * raw_class + args.shrinkage * np.broadcast_to(global_values, raw_class.shape)
    ) / (total_counts + args.shrinkage)
    payload = {
        "protocol": {
            "name": "CR-FMC-PC source-only cross-fitted reliability",
            "source_participants": list(SOURCE_PARTICIPANTS),
            "validation_participants_read": False,
            "final_test_participants_read": False,
            "final_test_accessed": False,
            "crossfit_folds": fold_protocols,
            "development_truncation": {
                "max_folds": args.max_folds,
                "max_holdout_batches": args.max_holdout_batches,
            },
            "teacher_training": "clean RCBD, fixed epoch, no held-out checkpoint selection",
            "retention": "exp(-max(CE_augmented-CE_clean,0)/temperature)",
            "pair_aggregation_during_student_training": "minimum of changed arm/leg reliability",
        },
        "source": {
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "temperature": args.temperature,
        "shrinkage_pseudocount": args.shrinkage,
        "severity_bin_edges_degrees": edges.tolist(),
        "severity_evaluation_degrees": centers.tolist(),
        "pair_names": list(PAIR_NAMES),
        "global_reliability": global_values.tolist(),
        "class_reliability": class_values.tolist(),
        "class_counts": total_counts.tolist(),
        "raw_class_reliability": raw_class.tolist(),
    }
    save_json(payload, output / "reliability_table.json")
    save_json(
        {
            "state": "completed",
            "folds": len(selected_folds),
            "classes": num_classes,
            "bins": len(centers),
            "final_test_accessed": False,
        },
        output / "STATUS.json",
    )
    (output / "COMPLETED").write_text("completed\n", encoding="utf-8")


if __name__ == "__main__":
    main()
