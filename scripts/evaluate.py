from __future__ import annotations

import argparse
import sys
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
    parser = argparse.ArgumentParser(description="Evaluate a saved RCBD-HAR checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--participant")
    parser.add_argument("--participants", nargs="+")
    parser.add_argument("--device")
    parser.add_argument("--skip-swap-stress", action="store_true")
    parser.add_argument("--output", default="outputs/evaluation.json")
    args = parser.parse_args()
    device = resolve_device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = state["config"]
    manifest = args.manifest or config["data"]["manifest"]
    frame = load_manifest(manifest)
    if args.participant and args.participants:
        raise ValueError("Use either --participant or --participants, not both")
    selected_participants = args.participants or (
        [str(args.participant)] if args.participant else None
    )
    if selected_participants:
        selected = {str(value) for value in selected_participants}
        frame = frame[frame.participant.astype(str).isin(selected)].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"No samples for participants {sorted(selected)}")
    dataset = WindowManifestDataset(
        frame,
        normalization=config["data"].get("normalization", "scale_only"),
        use_audio=bool(config["data"].get("use_audio", False)),
        limb_mode=config["data"].get("limb_mode", "bilateral"),
        normalization_stats=config["data"].get("normalization_stats"),
        normalization_scale=float(config["data"].get("normalization_scale", 8.0)),
        missing_strategy=config["data"].get("missing_strategy", "error"),
    )
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    model = build_model(config).to(device)
    model.load_state_dict(state["model"])
    metrics = evaluate(
        model,
        loader,
        device,
        int(config["data"]["num_classes"]),
        not args.skip_swap_stress,
        config.get("evaluation"),
    )
    save_json(metrics, args.output)
    print(f"macro-F1={metrics['macro_f1']:.4f}; results={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
