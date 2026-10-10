from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcbd_har.challenge_features import (  # noqa: E402
    DEFAULT_SCALES_SECONDS,
    LIMB_ORDER,
    extract_multiresolution_features,
    feature_names,
    repair_recording_from_validity,
)
from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.utils import participant_sort_key, save_json  # noqa: E402


def _single_path(rows, column: str) -> str:
    paths = rows[column].drop_duplicates().tolist()
    if len(paths) != 1:
        raise ValueError(f"Expected one {column} per participant, got {len(paths)}")
    return str(paths[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract WEAR Challenge 2024 multi-scale features")
    parser.add_argument("--manifest", default="data/wear/manifest.csv")
    parser.add_argument("--output", default="data/wear/challenge_features")
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--scales", nargs="+", type=int, default=list(DEFAULT_SCALES_SECONDS))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--missing-strategy", choices=["zero", "contralateral"], default="contralateral"
    )
    parser.add_argument("--participants", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    frame = load_manifest(args.manifest)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    participants = sorted(
        frame.participant.astype(str).unique().tolist(), key=participant_sort_key
    )
    if args.participants:
        requested = {str(value) for value in args.participants}
        unknown = requested.difference(participants)
        if unknown:
            raise ValueError(f"Unknown participants: {sorted(unknown)}")
        participants = [value for value in participants if value in requested]

    save_json(feature_names(args.scales), output / "feature_names.json")
    metadata = {
        "manifest": str(Path(args.manifest).resolve()),
        "sample_rate": args.sample_rate,
        "scales_seconds": args.scales,
        "missing_strategy": args.missing_strategy,
        "feature_count": len(feature_names(args.scales)),
        "participants": participants,
    }
    save_json(metadata, output / "metadata.json")

    for participant in participants:
        destination = output / f"{participant}.npz"
        if destination.exists() and not args.overwrite:
            print(f"skip {participant}: {destination} exists", flush=True)
            continue
        started = time.time()
        rows = frame[frame.participant == participant].sort_values("center", kind="stable")
        arrays = {
            limb: np.load(_single_path(rows, f"{limb}_path"), mmap_mode="r", allow_pickle=False)
            for limb in LIMB_ORDER
        }
        validity = {}
        for limb in LIMB_ORDER:
            column = f"{limb}_validity_path"
            if column in rows:
                validity[limb] = np.load(
                    _single_path(rows, column), mmap_mode="r", allow_pickle=False
                ).astype(bool)
        repaired = repair_recording_from_validity(
            arrays, validity, strategy=args.missing_strategy
        )
        features = extract_multiresolution_features(
            repaired,
            rows.center.to_numpy(dtype=np.int64),
            sample_rate=args.sample_rate,
            scales_seconds=args.scales,
            batch_size=args.batch_size,
        )
        np.savez_compressed(
            destination,
            features=features,
            labels=rows.label.to_numpy(dtype=np.int64),
            centers=rows.center.to_numpy(dtype=np.int64),
            row_indices=rows.index.to_numpy(dtype=np.int64),
        )
        elapsed = time.time() - started
        nan_fraction = float(np.isnan(features).mean())
        print(
            f"saved {participant}: {features.shape}, nan={nan_fraction:.4%}, "
            f"elapsed={elapsed / 60:.1f} min",
            flush=True,
        )

    (output / "_SUCCESS").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
