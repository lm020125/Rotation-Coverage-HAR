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
from rcbd_har.engine import forward_batch  # noqa: E402
from rcbd_har.models import build_model  # noqa: E402
from rcbd_har.utils import move_batch, resolve_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export time-resolved RCBD role scores to CSV")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--participant")
    parser.add_argument("--device")
    parser.add_argument("--output", default="outputs/role_traces.csv")
    args = parser.parse_args()
    device = resolve_device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = state["config"]
    if config["model"].get("variant", "rcbd") != "rcbd":
        raise ValueError("Role traces only exist for the RCBD variant")
    frame = load_manifest(args.manifest or config["data"]["manifest"])
    if args.participant:
        frame = frame[frame.participant.astype(str) == str(args.participant)].reset_index(drop=True)
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
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            output = forward_batch(model, batch)
            role_keys = [
                key for key in output if key == "role_score" or key.endswith("_role_score")
            ]
            for role_key in role_keys:
                role = output[role_key].detach().cpu().numpy()
                signed = role.mean(axis=-1) if role.shape[-1] > 1 else role[..., 0]
                magnitude = np.abs(role).mean(axis=-1)
                pair = role_key.removesuffix("_role_score") or "bilateral"
                for sample_index, path in enumerate(raw_batch["path"]):
                    for time_index in range(role.shape[1]):
                        rows.append(
                            {
                                "path": path,
                                "participant": raw_batch["participant"][sample_index],
                                "label": int(raw_batch["label"][sample_index]),
                                "pair": pair,
                                "time_index": time_index,
                                "role_signed_mean": float(signed[sample_index, time_index]),
                                "role_abs_mean": float(magnitude[sample_index, time_index]),
                            }
                        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
