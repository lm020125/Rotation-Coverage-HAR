from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcbd_har.placement import estimate_placement_profile  # noqa: E402


LIMBS = ("left_wrist", "right_wrist", "left_ankle", "right_ankle")


def load_unlabeled_manifest(path: str | Path) -> pd.DataFrame:
    columns = [
        "participant",
        *[f"{limb}_path" for limb in LIMBS],
        *[f"{limb}_validity_path" for limb in LIMBS],
    ]
    frame = pd.read_csv(path, usecols=columns)
    frame["participant"] = frame["participant"].astype(str)
    return frame.drop_duplicates().reset_index(drop=True)


def unique_source(frame: pd.DataFrame, participant: str, column: str) -> str:
    values = frame.loc[frame["participant"].eq(participant), column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(
            f"Expected one {column} for {participant}, found {len(values)}"
        )
    return str(values.iloc[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate acceleration-identifiable Ideal-to-Self placement profiles "
            "without loading activity labels."
        )
    )
    parser.add_argument("--ideal-manifest", required=True)
    parser.add_argument("--self-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--participants", nargs="+", required=True)
    args = parser.parse_args()

    participants = [str(value) for value in args.participants]
    ideal = load_unlabeled_manifest(args.ideal_manifest)
    self_placed = load_unlabeled_manifest(args.self_manifest)
    payload: dict[str, Any] = {
        "protocol": {
            "name": "REALDISP unlabeled training-participant placement calibration",
            "labels_read": False,
            "participants": participants,
            "ideal_manifest": str(Path(args.ideal_manifest).resolve()),
            "self_manifest": str(Path(args.self_manifest).resolve()),
            "target_validation_participants_read": False,
            "identifiable_rotation": "minimum-angle gravity-direction tilt; yaw unobserved",
        },
        "limbs": {limb: [] for limb in LIMBS},
        "excluded": [],
    }
    for participant in participants:
        for limb in LIMBS:
            try:
                ideal_path = unique_source(ideal, participant, f"{limb}_path")
                self_path = unique_source(self_placed, participant, f"{limb}_path")
                ideal_validity_path = unique_source(
                    ideal, participant, f"{limb}_validity_path"
                )
                self_validity_path = unique_source(
                    self_placed, participant, f"{limb}_validity_path"
                )
                profile = estimate_placement_profile(
                    np.load(ideal_path, mmap_mode="r", allow_pickle=False),
                    np.load(self_path, mmap_mode="r", allow_pickle=False),
                    ideal_validity=np.load(
                        ideal_validity_path, mmap_mode="r", allow_pickle=False
                    ),
                    self_validity=np.load(
                        self_validity_path, mmap_mode="r", allow_pickle=False
                    ),
                )
                profile["participant"] = participant
                payload["limbs"][limb].append(profile)
            except ValueError as error:
                payload["excluded"].append(
                    {"participant": participant, "limb": limb, "reason": str(error)}
                )
    for limb, profiles in payload["limbs"].items():
        if len(profiles) < 2:
            raise RuntimeError(f"Too few valid placement profiles for {limb}: {len(profiles)}")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(
        f"wrote {sum(len(value) for value in payload['limbs'].values())} profiles; "
        f"excluded={len(payload['excluded'])}; output={output}"
    )


if __name__ == "__main__":
    main()
