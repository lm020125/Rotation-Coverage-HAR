from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcbd_har.challenge_features import LIMB_ORDER, repair_recording_from_validity  # noqa: E402
from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.relational_features import (  # noqa: E402
    DEFAULT_RELATIONAL_SCALES_SECONDS,
    extract_relational_features,
    relational_feature_names,
)
from rcbd_har.utils import participant_sort_key, save_json  # noqa: E402


def _single_path(rows, column: str) -> str:
    paths = rows[column].drop_duplicates().tolist()
    if len(paths) != 1:
        raise ValueError(f"Expected one {column}, got {len(paths)}")
    return str(paths[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract WEAR relational spectral features")
    parser.add_argument("--manifest", default="data/wear/manifest.csv")
    parser.add_argument("--output", default="data/wear/relational_features")
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--scales", nargs="+", type=int, default=list(DEFAULT_RELATIONAL_SCALES_SECONDS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--participants", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    frame = load_manifest(args.manifest)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    participants = sorted(frame.participant.astype(str).unique(), key=participant_sort_key)
    if args.participants:
        requested = set(args.participants)
        participants = [value for value in participants if value in requested]
        if len(participants) != len(requested):
            raise ValueError("At least one requested participant is unknown")
    limb_names, pair_names = relational_feature_names(args.scales)
    metadata = {
        "sample_rate": args.sample_rate,
        "scales_seconds": args.scales,
        "limb_feature_count": len(limb_names),
        "pair_feature_count": len(pair_names),
        "participants": participants,
    }
    save_json(metadata, output / "metadata.json")
    save_json({"limb": limb_names, "pair": pair_names}, output / "feature_names.json")

    for participant in participants:
        destination = output / f"{participant}.npz"
        if destination.exists() and not args.overwrite:
            print(f"skip {participant}: exists", flush=True)
            continue
        started = time.time()
        rows = frame[frame.participant == participant].sort_values("center", kind="stable")
        arrays = {
            limb: np.load(_single_path(rows, f"{limb}_path"), mmap_mode="r", allow_pickle=False)
            for limb in LIMB_ORDER
        }
        validity = {
            limb: np.load(
                _single_path(rows, f"{limb}_validity_path"), mmap_mode="r", allow_pickle=False
            ).astype(bool)
            for limb in LIMB_ORDER
        }
        repaired = repair_recording_from_validity(arrays, validity, strategy="contralateral")
        limb_features, pair_features = extract_relational_features(
            repaired,
            rows.center.to_numpy(np.int64),
            sample_rate=args.sample_rate,
            scales_seconds=args.scales,
            batch_size=args.batch_size,
        )
        np.savez_compressed(
            destination,
            limb_features=limb_features,
            pair_features=pair_features,
            labels=rows.label.to_numpy(np.int64),
            row_indices=rows.index.to_numpy(np.int64),
        )
        print(
            f"saved {participant}: limb={limb_features.shape}, pair={pair_features.shape}, "
            f"nan={np.isnan(limb_features).mean():.3%}/{np.isnan(pair_features).mean():.3%}, "
            f"elapsed={(time.time() - started) / 60:.1f} min",
            flush=True,
        )
    (output / "_SUCCESS").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

