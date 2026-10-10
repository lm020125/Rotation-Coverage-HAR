from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcbd_har.data import WindowManifestDataset, load_manifest  # noqa: E402
from rcbd_har.engine import (  # noqa: E402
    expand_probabilities_to_samples,
    predict_probabilities,
    smooth_probabilities,
)
from rcbd_har.metrics import classification_metrics  # noqa: E402
from rcbd_har.models import build_model  # noqa: E402
from rcbd_har.utils import resolve_device, save_json  # noqa: E402


def discover_checkpoints(explicit: list[str] | None, root: str | None) -> list[Path]:
    paths = [Path(value).resolve() for value in (explicit or [])]
    if root:
        paths.extend(
            path.resolve()
            for path in Path(root).resolve().rglob("best.pt")
            if "selection" not in path.parts
        )
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError("No final best.pt checkpoints were found")
    return paths


def validate_checkpoint_grid(paths: list[Path], allow_incomplete: bool) -> None:
    combinations = []
    for path in paths:
        seed_parts = [part for part in path.parts if part.startswith("seed_")]
        fold_parts = [part for part in path.parts if part.startswith("fold_")]
        if len(seed_parts) == 1 and len(fold_parts) == 1:
            combinations.append((seed_parts[0], fold_parts[0]))
    if not combinations or allow_incomplete:
        return
    seeds = sorted({seed for seed, _ in combinations})
    folds = sorted({fold for _, fold in combinations})
    expected = {(seed, fold) for seed in seeds for fold in folds}
    missing = sorted(expected.difference(combinations))
    if missing:
        raise ValueError(f"Incomplete seed/fold checkpoint grid: missing {missing}")


def dataset_from_config(frame: pd.DataFrame, config: dict) -> WindowManifestDataset:
    return WindowManifestDataset(
        frame,
        normalization=config["data"].get("normalization", "scale_only"),
        use_audio=bool(config["data"].get("use_audio", False)),
        limb_mode=config["data"].get("limb_mode", "bilateral"),
        normalization_stats=config["data"].get("normalization_stats"),
        normalization_scale=float(config["data"].get("normalization_scale", 8.0)),
        missing_strategy=config["data"].get("missing_strategy", "error"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average fold/seed probabilities on unknown WEAR data"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cv-root")
    parser.add_argument("--checkpoints", nargs="+")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", default="outputs/wear_test")
    parser.add_argument("--expand-samplewise", action="store_true")
    parser.add_argument("--allow-incomplete-grid", action="store_true")
    args = parser.parse_args()
    checkpoints = discover_checkpoints(args.checkpoints, args.cv_root)
    validate_checkpoint_grid(checkpoints, args.allow_incomplete_grid)
    frame = load_manifest(args.manifest, require_labels=False)
    device = resolve_device(args.device)
    probability_sets = []
    reference: dict | None = None
    for checkpoint in checkpoints:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        config = state["config"]
        signature = {
            "num_classes": int(config["data"]["num_classes"]),
            "limb_mode": config["data"].get("limb_mode", "bilateral"),
            "variant": config["model"].get("variant", "rcbd"),
        }
        if reference is None:
            reference = {"signature": signature, "config": config}
        elif signature != reference["signature"]:
            raise ValueError(f"Incompatible checkpoint configuration: {checkpoint}")
        dataset = dataset_from_config(frame, config)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size or int(config["training"]["batch_size"]),
            shuffle=False,
        )
        model = build_model(config).to(device)
        model.load_state_dict(state["model"])
        probability_sets.append(predict_probabilities(model, loader, device))
        print(f"predicted {checkpoint}")
    probabilities = np.mean(np.stack(probability_sets, axis=0), axis=0)
    assert reference is not None
    evaluation = reference["config"].get("evaluation", {})
    if bool(evaluation.get("enabled", False)):
        stride = float(evaluation.get("stride_seconds", 0.5))
        probabilities = smooth_probabilities(
            probabilities,
            frame,
            int(round(float(evaluation.get("radius_seconds", 5.0)) / stride)),
            float(evaluation.get("sigma_seconds", 3.0)) / stride,
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    predictions = probabilities.argmax(axis=-1)
    compressed = frame.copy()
    compressed["prediction"] = predictions
    compressed["confidence"] = probabilities.max(axis=-1)
    compressed.to_csv(output / "window_predictions.csv", index=False, encoding="utf-8")
    np.savez_compressed(output / "window_probabilities.npz", probabilities=probabilities)
    metadata = {
        "checkpoints": [str(path) for path in checkpoints],
        "models": len(checkpoints),
        "windows": len(frame),
    }
    if (frame["label"] >= 0).all():
        metadata["metrics"] = classification_metrics(
            frame["label"].to_numpy(),
            predictions,
            int(reference["signature"]["num_classes"]),
        )
    if args.expand_samplewise:
        sample_dir = output / "samplewise"
        sample_dir.mkdir(parents=True, exist_ok=True)
        expanded = expand_probabilities_to_samples(probabilities, frame)
        sample_targets: list[np.ndarray] = []
        sample_predictions: list[np.ndarray] = []
        for recording, values in expanded.items():
            np.savez_compressed(
                sample_dir / f"{recording}.npz",
                probabilities=values,
                predictions=values.argmax(axis=-1),
            )
            participant, recording_id = recording.split("__", maxsplit=1)
            recording_column = "recording_id" if "recording_id" in frame else "session"
            rows = frame[
                (frame["participant"].astype(str) == participant)
                & (frame[recording_column].astype(str) == recording_id)
            ]
            if "labels_path" in rows and len(rows):
                label_paths = rows["labels_path"].drop_duplicates().tolist()
                if len(label_paths) == 1:
                    labels = np.load(label_paths[0], allow_pickle=False)
                    if len(labels) != len(values):
                        raise ValueError(f"Label length mismatch for {recording}")
                    if (labels >= 0).all():
                        sample_targets.append(labels)
                        sample_predictions.append(values.argmax(axis=-1))
        if sample_targets:
            metadata["samplewise_metrics"] = classification_metrics(
                np.concatenate(sample_targets),
                np.concatenate(sample_predictions),
                int(reference["signature"]["num_classes"]),
            )
    save_json(metadata, output / "ensemble.json")
    print(f"ensemble predictions: {output.resolve()}")


if __name__ == "__main__":
    main()
