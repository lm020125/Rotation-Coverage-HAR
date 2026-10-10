from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LIMB_COLUMNS = ("left_wrist", "right_wrist", "left_ankle", "right_ankle")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a non-duplicating WEAR manifest from aligned continuous NPY arrays. "
            "The index CSV needs participant,session,left_wrist,right_wrist,left_ankle,"
            "right_ankle,labels. Limb arrays are [N,3], labels are [N]."
        )
    )
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", default="data/wear/manifest.csv")
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--context-seconds", type=float, default=64.0)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--unlabeled", action="store_true")
    args = parser.parse_args()
    index_path = Path(args.index).resolve()
    source = pd.read_csv(index_path)
    required = {"participant", "session", *LIMB_COLUMNS}
    if not args.unlabeled:
        required.add("labels")
    if missing := required.difference(source.columns):
        raise ValueError(f"Index CSV lacks {sorted(missing)}")
    context = int(round(args.context_seconds * args.sample_rate))
    stride = int(round(args.stride_seconds * args.sample_rate))
    if context <= 0 or stride <= 0:
        raise ValueError("context and stride must be positive")
    left_context = context // 2
    right_context = context - left_context
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for _, record in source.iterrows():
        path_columns = list(LIMB_COLUMNS) + ([] if args.unlabeled else ["labels"])
        validity_columns = [f"{limb}_validity" for limb in LIMB_COLUMNS]
        present_validity = [column for column in validity_columns if column in source.columns]
        if present_validity and len(present_validity) != len(validity_columns):
            raise ValueError("Provide either all four *_validity columns or none")
        path_columns += present_validity
        resolved = {
            column: (index_path.parent / str(record[column])).resolve() for column in path_columns
        }
        arrays = {
            column: np.load(path, mmap_mode="r", allow_pickle=False)
            for column, path in resolved.items()
        }
        lengths = {column: len(value) for column, value in arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Unaligned session {record.session}: {lengths}")
        for limb in LIMB_COLUMNS:
            if arrays[limb].ndim != 2 or arrays[limb].shape[1] != 3:
                raise ValueError(f"{limb} must be [N,3], got {arrays[limb].shape}")
        labels = (
            np.full(next(iter(lengths.values())), -1, dtype=np.int64)
            if args.unlabeled
            else arrays["labels"]
        )
        if labels.ndim != 1:
            raise ValueError(f"labels must be [N], got {labels.shape}")
        for column in present_validity:
            if arrays[column].shape != arrays[column.removesuffix("_validity")].shape:
                raise ValueError(f"{column} shape must match its sensor array")
        for center in range(0, len(labels), stride):
            rows.append(
                {
                    "participant": str(record.participant),
                    "session": str(record.session),
                    "recording_id": f"{record.participant}__{record.session}",
                    "center": center,
                    "num_samples": len(labels),
                    "sample_rate": args.sample_rate,
                    **({"labels_path": str(resolved["labels"])} if not args.unlabeled else {}),
                    "start": center - left_context,
                    "stop": center + right_context,
                    "label": int(labels[center]),
                    "left_wrist_path": str(resolved["left_wrist"]),
                    "right_wrist_path": str(resolved["right_wrist"]),
                    "left_ankle_path": str(resolved["left_ankle"]),
                    "right_ankle_path": str(resolved["right_ankle"]),
                    **{
                        f"{limb}_validity_path": str(resolved[f"{limb}_validity"])
                        for limb in LIMB_COLUMNS
                        if f"{limb}_validity" in resolved
                    },
                }
            )
    frame = pd.DataFrame(rows).sort_values(["participant", "session", "center"])
    frame.to_csv(output_path, index=False, encoding="utf-8")
    print(
        f"wrote {len(frame)} prediction windows ({args.context_seconds:g}s context, "
        f"{args.stride_seconds:g}s stride) -> {output_path}"
    )


if __name__ == "__main__":
    main()
