from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcbd_har.config import load_config, with_overrides  # noqa: E402
from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.experiment import run_fold  # noqa: E402
from rcbd_har.utils import save_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-free LOPO evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=["rcbd", "concat", "mean", "deepsets"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--participants", nargs="*", help="Optional subset for debugging")
    parser.add_argument("--device")
    parser.add_argument("--output", default="outputs/lopo")
    args = parser.parse_args()
    config = with_overrides(
        load_config(args.config),
        **{"model.variant": args.variant, "training.epochs": args.epochs},
    )
    frame = load_manifest(config["data"]["manifest"])
    available = sorted(frame.participant.astype(str).unique().tolist())
    participants = args.participants or available
    unknown = sorted(set(participants).difference(available))
    if unknown:
        raise ValueError(f"Unknown participants: {unknown}")
    root = Path(args.output)
    results = []
    for participant in participants:
        print(f"\n=== LOPO test participant: {participant} ===")
        results.append(
            run_fold(
                config,
                test_participant=participant,
                output_dir=root / f"test_{participant}",
                device_name=args.device,
            )
        )
    summary = {
        "folds": len(results),
        "participants": participants,
        "macro_f1_mean": float(np.mean([r["macro_f1"] for r in results])),
        "macro_f1_std": float(np.std([r["macro_f1"] for r in results])),
        "accuracy_mean": float(np.mean([r["accuracy"] for r in results])),
        "swap_f1_delta_mean": float(np.mean([r["swap_stress"]["macro_f1_delta"] for r in results])),
        "per_fold": [
            {
                "participant": r["test_participant"],
                "macro_f1": r["macro_f1"],
                "accuracy": r["accuracy"],
                "swap_stress": r["swap_stress"],
            }
            for r in results
        ],
    }
    save_json(summary, root / "summary.json")
    print(f"\nLOPO macro-F1: {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}")


if __name__ == "__main__":
    main()
