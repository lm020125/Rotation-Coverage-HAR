from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_temporal_decoder_cv import evaluate_labels  # noqa: E402
from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.residual_expert import conservative_residual_majority  # noqa: E402
from rcbd_har.utils import save_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed conservative majority ensemble of residual gate seeds"
    )
    parser.add_argument("--manifest", default="data/wear/manifest.csv")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = load_manifest(args.manifest).reset_index(drop=True)
    archives = [np.load(Path(value).resolve() / "oof_predictions.npz") for value in args.inputs]
    reference = archives[0]
    required = (
        "row_indices",
        "targets",
        "main_predictions",
        "expert_predictions",
    )
    for key in required:
        if any(not np.array_equal(reference[key], archive[key]) for archive in archives[1:]):
            raise ValueError(f"Residual seed archives do not share {key}")
    rows = reference["row_indices"].astype(np.int64)
    if not np.array_equal(rows, np.arange(len(frame))):
        raise ValueError("Residual seed ensemble requires complete manifest-ordered OOF")
    main_predictions = reference["main_predictions"].astype(np.int64)
    expert_predictions = reference["expert_predictions"].astype(np.int64)
    members = np.stack(
        [archive["gated_predictions"].astype(np.int64) for archive in archives]
    )
    predictions = conservative_residual_majority(
        main_predictions, expert_predictions, members
    )
    num_classes = int(frame.label.max()) + 1
    overall = evaluate_labels(predictions, frame, num_classes)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "Fixed conservative strict-majority residual gate seed ensemble",
        "strict_outer_test_label_free": True,
        "tie_policy": "retain unchanged main prediction",
        "inputs": args.inputs,
        "member_count": len(archives),
        "expert_switch_fraction": float(np.mean(predictions != main_predictions)),
        "overall": overall,
    }
    save_json(summary, output / "summary.json")
    np.savez_compressed(
        output / "oof_predictions.npz",
        row_indices=rows,
        main_predictions=main_predictions,
        expert_predictions=expert_predictions,
        member_predictions=members,
        ensemble_predictions=predictions,
        targets=reference["targets"].astype(np.int64),
    )
    print(
        f"overall center={overall['center']['macro_f1']:.4%}; "
        f"sample={overall['samplewise']['macro_f1']:.4%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
