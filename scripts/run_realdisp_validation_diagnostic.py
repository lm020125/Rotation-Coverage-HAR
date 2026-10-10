from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcbd_har.config import load_config, with_overrides  # noqa: E402
from rcbd_har.data import (  # noqa: E402
    WindowManifestDataset,
    estimate_global_normalization,
    load_manifest,
)
from rcbd_har.engine import evaluate, evaluate_pair_rotation_stress, fit  # noqa: E402
from rcbd_har.models import build_model  # noqa: E402
from rcbd_har.utils import resolve_device, save_json, seed_everything  # noqa: E402


DEFAULT_TRAIN = [f"subject{value}" for value in range(1, 11)]
DEFAULT_VAL = ["subject11", "subject12"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train on REALDISP subjects 1-10 and evaluate only the development "
            "subjects 11-12, including deterministic one-side rotation stress."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--stress-repeats", type=int, default=3)
    parser.add_argument("--stress-max-degrees", type=float, default=90.0)
    parser.add_argument(
        "--placement-profile",
        help=(
            "Audited unlabeled training-participant placement profile JSON. "
            "When set, it replaces arbitrary SO(3) quality augmentation."
        ),
    )
    parser.add_argument("--train-participants", nargs="+", default=DEFAULT_TRAIN)
    parser.add_argument("--val-participants", nargs="+", default=DEFAULT_VAL)
    args = parser.parse_args()

    if args.stress_repeats < 1:
        raise ValueError("stress-repeats must be positive")
    config = with_overrides(
        load_config(args.config),
        seed=args.seed,
        **{
            "training.epochs": args.epochs,
            "training.batch_size": args.batch_size,
            "training.num_workers": args.num_workers,
            "training.quality_augmentation_profile_path": args.placement_profile,
        },
    )
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = resolve_device(args.device)
    frame = load_manifest(config["data"]["manifest"])
    participant_series = frame["participant"].astype(str)
    train_participants = [str(value) for value in args.train_participants]
    val_participants = [str(value) for value in args.val_participants]
    if set(train_participants) & set(val_participants):
        raise ValueError("Training and validation participants must be disjoint")
    available = set(participant_series.unique())
    requested = set(train_participants + val_participants)
    if missing := requested - available:
        raise ValueError(f"Participants are absent from manifest: {sorted(missing)}")
    train_indices = frame.index[participant_series.isin(train_participants)].tolist()
    val_indices = frame.index[participant_series.isin(val_participants)].tolist()
    if not train_indices or not val_indices:
        raise ValueError("Validation diagnostic produced an empty partition")

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
    train_dataset = WindowManifestDataset(frame, train_indices, **dataset_kwargs)
    val_dataset = WindowManifestDataset(frame, val_indices, **dataset_kwargs)
    workers = int(config["training"].get("num_workers", 0))
    loader_kwargs = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(config, output / "resolved_config.json")
    save_json(
        {
            "name": "REALDISP validation-only FAPC diagnostic",
            "train_participants": train_participants,
            "validation_participants": val_participants,
            "excluded_final_test_participants": [f"subject{value}" for value in range(13, 18)],
            "final_test_accessed": False,
            "seed": seed,
            "sizes": {"train": len(train_dataset), "validation": len(val_dataset)},
        },
        output / "protocol.json",
    )
    model = build_model(config).to(device)
    model, history = fit(model, train_loader, val_loader, config, device, output)
    save_json(history, output / "history.json")
    clean = evaluate(
        model,
        val_loader,
        device,
        int(config["data"]["num_classes"]),
        swap_stress=True,
        smoothing=config.get("evaluation"),
    )
    stress = [
        evaluate_pair_rotation_stress(
            model,
            val_loader,
            device,
            int(config["data"]["num_classes"]),
            max_degrees=args.stress_max_degrees,
            seed=seed + 50_000 + repeat,
            smoothing=config.get("evaluation"),
        )
        for repeat in range(args.stress_repeats)
    ]
    stress_macro_f1 = np.asarray([item["macro_f1"] for item in stress], dtype=float)
    stress_accuracy = np.asarray([item["accuracy"] for item in stress], dtype=float)
    metrics = {
        "clean": clean,
        "rotation_stress": {
            "repeats": stress,
            "macro_f1_mean": float(stress_macro_f1.mean()),
            "macro_f1_std": float(stress_macro_f1.std(ddof=0)),
            "accuracy_mean": float(stress_accuracy.mean()),
            "accuracy_std": float(stress_accuracy.std(ddof=0)),
            "max_degrees": float(args.stress_max_degrees),
        },
        "final_test_accessed": False,
    }
    save_json(metrics, output / "validation_metrics.json")
    print(
        f"clean macro-F1={clean['macro_f1']:.6%}; "
        f"rotation-stress macro-F1={stress_macro_f1.mean():.6%} "
        f"+/- {stress_macro_f1.std(ddof=0):.6%}; output={output.resolve()}"
    )


if __name__ == "__main__":
    main()
