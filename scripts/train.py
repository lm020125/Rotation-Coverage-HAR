from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcbd_har.config import load_config, with_overrides  # noqa: E402
from rcbd_har.experiment import run_fold  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one participant-held-out RCBD-HAR fold")
    parser.add_argument("--config", required=True)
    parser.add_argument("--test-participant", required=True)
    parser.add_argument("--val-participant")
    parser.add_argument("--variant", choices=["rcbd", "concat", "mean", "deepsets"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output", default="outputs/single_fold")
    args = parser.parse_args()
    config = with_overrides(
        load_config(args.config),
        **{"model.variant": args.variant, "training.epochs": args.epochs},
    )
    metrics = run_fold(
        config,
        test_participant=args.test_participant,
        val_participant=args.val_participant,
        output_dir=args.output,
        device_name=args.device,
    )
    print(f"macro-F1={metrics['macro_f1']:.4f}, accuracy={metrics['accuracy']:.4f}")
    print(f"results: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
