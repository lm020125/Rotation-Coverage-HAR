from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcbd_har.data import WindowManifestDataset, load_manifest  # noqa: E402
from rcbd_har.engine import evaluate  # noqa: E402
from rcbd_har.models import build_model  # noqa: E402
from rcbd_har.utils import resolve_device, save_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained REALDISP checkpoint on another placement scenario "
            "without retraining or recalculating normalization statistics."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    parser.add_argument("--participants", nargs="+")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--quality-mode",
        choices=("learned", "fixed_one", "fixed_zero"),
        help=(
            "Inference-only QMBD quality-gate ablation. The checkpoint is loaded "
            "with its trained architecture before the runtime mode is overridden."
        ),
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in state or "config" not in state:
        raise ValueError(f"Not an RCBD-HAR checkpoint: {checkpoint_path}")
    config = deepcopy(state["config"])
    training_manifest = str(config["data"]["manifest"])
    normalization_stats = config["data"].get("normalization_stats")
    normalization = config["data"].get("normalization", "scale_only")
    if normalization in {"train_global", "train_rms_shared"} and not normalization_stats:
        raise ValueError("Checkpoint lacks train-only normalization statistics")

    manifest_path = Path(args.manifest).resolve()
    config["data"]["manifest"] = str(manifest_path)
    frame = load_manifest(manifest_path)
    if args.participants:
        participants = [str(value) for value in args.participants]
    else:
        split_path = checkpoint_path.parent / "split.json"
        if not split_path.exists():
            raise FileNotFoundError(
                "Provide --participants because split.json is absent beside the checkpoint"
            )
        participants = [
            str(value)
            for value in json.loads(split_path.read_text(encoding="utf-8"))["test_participants"]
        ]
    available = set(frame["participant"].astype(str).unique())
    missing = sorted(set(participants) - available)
    if missing:
        raise ValueError(f"Evaluation participants absent from manifest: {missing}")
    indices = frame.index[frame["participant"].astype(str).isin(participants)].tolist()
    if not indices:
        raise ValueError("Evaluation split is empty")

    dataset = WindowManifestDataset(
        frame,
        indices,
        normalization=normalization,
        use_audio=bool(config["data"].get("use_audio", False)),
        limb_mode=config["data"].get("limb_mode", "bilateral"),
        normalization_stats=normalization_stats,
        normalization_scale=float(config["data"].get("normalization_scale", 8.0)),
        missing_strategy=config["data"].get("missing_strategy", "error"),
    )
    device = resolve_device(args.device)
    workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else int(config["training"].get("num_workers", 0))
    )
    loader = DataLoader(
        dataset,
        batch_size=(
            int(args.batch_size)
            if args.batch_size is not None
            else int(config["training"]["batch_size"])
        ),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    model = build_model(config).to(device)
    model.load_state_dict(state["model"])
    overridden_quality_pairs = 0
    if args.quality_mode is not None:
        for module in model.modules():
            if getattr(module, "variant", None) == "qmbd" and hasattr(
                module, "quality_mode"
            ):
                module.quality_mode = args.quality_mode
                overridden_quality_pairs += 1
        if overridden_quality_pairs == 0:
            raise ValueError("--quality-mode requires a QMBD checkpoint")
    metrics = evaluate(
        model,
        loader,
        device,
        int(config["data"]["num_classes"]),
        swap_stress=True,
        smoothing=config.get("evaluation"),
    )
    metrics["evaluation_protocol"] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(state.get("epoch", -1)),
        "model_architecture": str(config["model"].get("architecture", "rcbd_har")),
        "model_variant": str(config["model"]["variant"]),
        "training_manifest": training_manifest,
        "evaluation_manifest": str(manifest_path),
        "participants": participants,
        "normalization_source": "checkpoint_train_split",
        "num_windows": len(dataset),
        "quality_mode_override": args.quality_mode,
        "quality_pairs_overridden": overridden_quality_pairs,
    }
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_json(metrics, output / "metrics.json")
    save_json(metrics["evaluation_protocol"], output / "protocol.json")
    print(
        f"macro-F1={metrics['macro_f1']:.6%}, accuracy={metrics['accuracy']:.6%}, "
        f"windows={len(dataset)}, output={output}"
    )


if __name__ == "__main__":
    main()
