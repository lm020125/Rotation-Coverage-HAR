from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert already-windowed bilateral arrays to RCBD NPZ files. "
            "The input index CSV needs participant,left,right,labels columns; each path points to "
            "an array [N,T,C] or labels [N]. This adapter is suitable for WEAR preprocessing outputs."
        )
    )
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", default="data/wear")
    args = parser.parse_args()
    index_path = Path(args.index).resolve()
    table = pd.read_csv(index_path)
    required = {"participant", "left", "right", "labels"}
    if missing := required.difference(table.columns):
        raise ValueError(f"Index CSV lacks {sorted(missing)}")
    output = Path(args.output)
    window_dir = output / "windows"
    window_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in table.iterrows():
        root = index_path.parent

        def load(p: str) -> np.ndarray:
            return np.load(root / str(p), allow_pickle=False)

        left, right, labels = load(row.left), load(row.right), load(row.labels)
        if left.ndim != 3 or right.shape != left.shape or labels.shape[0] != left.shape[0]:
            raise ValueError(
                f"Expected equal left/right [N,T,C] and labels [N] for {row.participant}; "
                f"got {left.shape}, {right.shape}, {labels.shape}"
            )
        for index in range(left.shape[0]):
            path = window_dir / f"{row.participant}_{index:06d}.npz"
            np.savez_compressed(
                path, left=left[index].astype(np.float32), right=right[index].astype(np.float32)
            )
            rows.append(
                {
                    "path": str(path.relative_to(output)),
                    "participant": str(row.participant),
                    "label": int(labels[index]),
                }
            )
    pd.DataFrame(rows).to_csv(output / "manifest.csv", index=False, encoding="utf-8")
    print(f"converted {len(rows)} windows -> {(output / 'manifest.csv').resolve()}")


if __name__ == "__main__":
    main()
