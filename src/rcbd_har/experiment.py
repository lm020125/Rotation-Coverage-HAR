from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from .data import (
    WindowManifestDataset,
    build_participant_split,
    estimate_global_normalization,
    load_manifest,
)
from .engine import evaluate, fit
from .models import build_model
from .utils import resolve_device, save_json, seed_everything


def run_fold(
    config: dict[str, Any],
    test_participant: str,
    output_dir: str | Path,
    val_participant: str | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    frame = load_manifest(config["data"]["manifest"])
    split = build_participant_split(frame, test_participant, val_participant)
    return run_group_fold(
        config,
        test_participants=[split.test_participant],
        val_participants=[split.val_participant],
        output_dir=output_dir,
        device_name=device_name,
    )


def run_group_fold(
    config: dict[str, Any],
    test_participants: list[str],
    val_participants: list[str],
    output_dir: str | Path,
    device_name: str | None = None,
    refit_full_train: bool = False,
) -> dict[str, Any]:
    """Train one participant-group fold and evaluate its held-out participants."""
    config = deepcopy(config)
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(device_name)
    frame = load_manifest(config["data"]["manifest"])
    participants = frame["participant"].astype(str)
    test_participants = [str(value) for value in test_participants]
    val_participants = [str(value) for value in val_participants]
    if set(test_participants) & set(val_participants):
        raise ValueError("Validation and test participant groups must be disjoint")
    test_indices = frame.index[participants.isin(test_participants)].tolist()
    val_indices = frame.index[participants.isin(val_participants)].tolist()
    train_indices = frame.index[~participants.isin(test_participants + val_participants)].tolist()
    if not train_indices or not val_indices or not test_indices:
        raise ValueError("Grouped split produced an empty train, validation, or test partition")
    normalization = config["data"].get("normalization", "scale_only")
    if normalization in {"train_global", "train_rms_shared"}:
        config["data"]["normalization_stats"] = estimate_global_normalization(
            frame.iloc[train_indices],
            config["data"].get("limb_mode", "bilateral"),
            mode=normalization,
        )
    dataset_kwargs = {
        "normalization": normalization,
        "use_audio": bool(config["data"].get("use_audio", False)),
        "limb_mode": config["data"].get("limb_mode", "bilateral"),
        "normalization_stats": config["data"].get("normalization_stats"),
        "normalization_scale": float(config["data"].get("normalization_scale", 8.0)),
        "missing_strategy": config["data"].get("missing_strategy", "error"),
    }
    train_dataset = WindowManifestDataset(
        frame,
        train_indices,
        swap_probability=float(config["training"].get("swap_augmentation_probability", 0.0)),
        **dataset_kwargs,
    )
    val_dataset = WindowManifestDataset(frame, val_indices, **dataset_kwargs)
    test_dataset = WindowManifestDataset(frame, test_indices, **dataset_kwargs)
    workers = int(config["training"].get("num_workers", 0))
    loader_kwargs = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config["data"].get("normalization_stats"):
        save_json(config["data"]["normalization_stats"], output_dir / "normalization.json")
    save_json(
        {
            "test_participants": test_participants,
            "val_participants": val_participants,
            "train_participants": sorted(train_dataset.frame.participant.unique().tolist()),
            "sizes": {
                "train": len(train_dataset),
                "val": len(val_dataset),
                "test": len(test_dataset),
            },
        },
        output_dir / "split.json",
    )
    model = build_model(config).to(device)
    selection_dir = output_dir / "selection" if refit_full_train else output_dir
    model, selection_history = fit(model, train_loader, val_loader, config, device, selection_dir)
    history: dict[str, Any] | list[dict[str, float]] = selection_history
    if refit_full_train:
        selected_epoch = int(
            max(
                selection_history,
                key=lambda item: item.get("val_selection_score", item["val_macro_f1"]),
            )["epoch"]
        )
        refit_indices = sorted(train_indices + val_indices)
        if normalization in {"train_global", "train_rms_shared"}:
            config["data"]["normalization_stats"] = estimate_global_normalization(
                frame.iloc[refit_indices],
                config["data"].get("limb_mode", "bilateral"),
                mode=normalization,
            )
        refit_kwargs = {
            **dataset_kwargs,
            "normalization_stats": config["data"].get("normalization_stats"),
        }
        refit_dataset = WindowManifestDataset(
            frame,
            refit_indices,
            swap_probability=float(config["training"].get("swap_augmentation_probability", 0.0)),
            **refit_kwargs,
        )
        test_dataset = WindowManifestDataset(frame, test_indices, **refit_kwargs)
        refit_loader = DataLoader(refit_dataset, shuffle=True, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
        seed_everything(int(config.get("seed", 42)) + 10_000)
        model = build_model(config).to(device)
        model, refit_history = fit(
            model,
            refit_loader,
            None,
            config,
            device,
            output_dir,
            fixed_epochs=selected_epoch,
        )
        train_dataset = refit_dataset
        history = {
            "selected_epoch": selected_epoch,
            "selection": selection_history,
            "refit": refit_history,
        }
        if config["data"].get("normalization_stats"):
            save_json(config["data"]["normalization_stats"], output_dir / "normalization.json")
        save_json(
            {
                "test_participants": test_participants,
                "val_participants_for_epoch_selection": val_participants,
                "train_participants": sorted(train_dataset.frame.participant.unique().tolist()),
                "refit_full_train": True,
                "selected_epoch": selected_epoch,
                "sizes": {
                    "train": len(train_dataset),
                    "selection_val": len(val_dataset),
                    "test": len(test_dataset),
                },
            },
            output_dir / "split.json",
        )
    metrics = evaluate(
        model,
        test_loader,
        device,
        int(config["data"]["num_classes"]),
        swap_stress=True,
        smoothing=config.get("evaluation"),
    )
    metrics["test_participants"] = test_participants
    metrics["val_participants"] = val_participants
    if len(test_participants) == 1:
        metrics["test_participant"] = test_participants[0]
    if len(val_participants) == 1:
        metrics["val_participant"] = val_participants[0]
    save_json(history, output_dir / "history.json")
    save_json(metrics, output_dir / "metrics.json")
    return metrics
