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
        raise ValueError(
            f"Expected one {column} for {rows.participant.iloc[0]}, got {len(paths)}"
        )
    return str(paths[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the public WEAR-Challenge multiresolution statistical feature family "
            "from one REALDISP scenario manifest. Run separately for Ideal and Self."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument(
        "--scales", nargs="+", type=int, default=list(DEFAULT_SCALES_SECONDS)
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--missing-strategy", choices=["zero", "contralateral"], default="contralateral"
    )
    parser.add_argument("--participants", nargs="+")
    parser.add_argument(
        "--max-windows",
        type=int,
        help="Development-only cap per participant; omitted for paper experiments.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_windows is not None and args.max_windows <= 0:
        raise ValueError("--max-windows must be positive")
    if not args.scales or any(value <= 0 for value in args.scales):
        raise ValueError("--scales must contain positive durations")

    frame = load_manifest(args.manifest)
    scenarios = sorted(frame.scenario.astype(str).unique().tolist())
    if len(scenarios) != 1:
        raise ValueError(
            "Feature extraction requires a single-scenario manifest; run Ideal and Self "
            f"separately, got {scenarios}"
        )
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

    names = feature_names(args.scales)
    save_json(names, output / "feature_names.json")
    metadata = {
        "source": "WEAR Challenge 2024 multiresolution statistical feature family",
        "dataset": "REALDISP",
        "scenario": scenarios[0],
        "manifest": str(Path(args.manifest).resolve()),
        "sample_rate": args.sample_rate,
        "scales_seconds": args.scales,
        "missing_strategy": args.missing_strategy,
        "feature_count": len(names),
        "participants": participants,
        "max_windows": args.max_windows,
    }
    save_json(metadata, output / "metadata.json")

    for participant in participants:
        destination = output / f"{participant}.npz"
        if destination.exists() and not args.overwrite:
            print(f"skip {participant}: {destination} exists", flush=True)
            continue
        started = time.time()
        rows = frame[frame.participant.astype(str) == participant].sort_values(
            ["recording_id", "center"], kind="stable"
        )
        recordings = rows.recording_id.astype(str).unique().tolist()
        if len(recordings) != 1:
            raise ValueError(
                f"Expected one {scenarios[0]} recording for {participant}, got {recordings}"
            )
        if args.max_windows is not None:
            rows = rows.iloc[: args.max_windows]
        arrays = {
            limb: np.load(
                _single_path(rows, f"{limb}_path"), mmap_mode="r", allow_pickle=False
            )
            for limb in LIMB_ORDER
        }
        validity: dict[str, np.ndarray] = {}
        for limb in LIMB_ORDER:
            column = f"{limb}_validity_path"
            if column in rows.columns:
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
        print(
            f"saved {participant}: shape={features.shape}, "
            f"missing-feature fraction={np.isnan(features).mean():.4%}, "
            f"elapsed={(time.time() - started) / 60:.1f} min",
            flush=True,
        )

    (output / "_SUCCESS").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
