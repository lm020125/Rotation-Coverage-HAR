from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HARMES_CLASSES = [
    "Null",
    "Floor cleaning",
    "Making tea",
    "Vacuum Cleaning",
    "Window cleaning",
    "Brushing teeth",
    "Washing dishes",
    "Cutting vegetables",
    "Putting away the dishes",
    "Cleaning out the dishwasher",
    "Watering plant",
    "Cleaning table",
    "Washing hands",
    "Drinking",
    "Cream hands",
    "Disinfecting hands",
]


def _normalise_label(label: Any) -> str:
    return " ".join(str(label).strip().lower().replace("_", " ").split())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a HARMES part_datas pickle into the RCBD window format. "
            "part_datas is the dictionary returned by the official get_part_datas()."
        )
    )
    parser.add_argument(
        "--input", required=True, help="Pickle containing participant -> [L,R,audio,H,label]"
    )
    parser.add_argument("--output", default="data/harmes")
    parser.add_argument("--without-audio", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    windows = output / "windows"
    windows.mkdir(parents=True, exist_ok=True)
    with Path(args.input).open("rb") as handle:
        part_datas = pickle.load(handle)
    if not isinstance(part_datas, dict):
        raise TypeError("Expected the official participant dictionary")
    canonical = {_normalise_label(name): idx for idx, name in enumerate(HARMES_CLASSES)}
    rows: list[dict[str, object]] = []
    for participant, values in sorted(part_datas.items(), key=lambda pair: str(pair[0])):
        if not isinstance(values, (list, tuple)) or len(values) < 5:
            raise ValueError(f"Participant {participant} is not [L,R,audio,H,label]")
        left_windows, right_windows, audio_windows, _, labels = values[:5]
        lengths = [len(left_windows), len(right_windows), len(labels)]
        if not args.without_audio:
            lengths.append(len(audio_windows))
        if len(set(lengths)) != 1:
            raise ValueError(f"Unaligned HARMES windows for {participant}: {lengths}")
        for index, label_name in enumerate(labels):
            key = _normalise_label(label_name)
            if key not in canonical:
                raise ValueError(
                    f"Unknown HARMES label {label_name!r}; update HARMES_CLASSES if needed"
                )
            left = np.asarray(left_windows[index], dtype=np.float32)
            right = np.asarray(right_windows[index], dtype=np.float32)
            if left.shape != right.shape or left.ndim != 2:
                raise ValueError(
                    f"Invalid bilateral shape at {participant}/{index}: {left.shape}/{right.shape}"
                )
            path = windows / f"{participant}_{index:06d}.npz"
            if args.without_audio:
                np.savez_compressed(path, left=left, right=right)
            else:
                audio = np.asarray(audio_windows[index], dtype=np.float32)
                if audio.ndim == 3 and audio.shape[0] == 1:
                    audio = audio[0]
                if audio.ndim != 2:
                    raise ValueError(f"Expected log-mel [M,F], got {audio.shape}")
                np.savez_compressed(path, left=left, right=right, audio=audio)
            rows.append(
                {
                    "path": str(path.relative_to(output)),
                    "participant": str(participant),
                    "label": canonical[key],
                    "label_name": HARMES_CLASSES[canonical[key]],
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "manifest.csv", index=False, encoding="utf-8")
    with (output / "classes.json").open("w", encoding="utf-8") as handle:
        json.dump(HARMES_CLASSES, handle, ensure_ascii=False, indent=2)
    print(f"converted {len(rows)} windows -> {(output / 'manifest.csv').resolve()}")


if __name__ == "__main__":
    main()
