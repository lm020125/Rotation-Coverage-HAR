from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcbd_har.config import load_config, with_overrides  # noqa: E402
from rcbd_har.data import load_manifest  # noqa: E402
from rcbd_har.experiment import run_group_fold  # noqa: E402
from rcbd_har.utils import participant_sort_key  # noqa: E402


DEFAULT_TRAIN = [f"subject{value}" for value in range(1, 11)]
DEFAULT_VAL = ["subject11", "subject12"]
DEFAULT_TEST = [f"subject{value}" for value in range(13, 18)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the published REALDISP user-independent split: subjects 1-10 train, "
            "11-12 validation, and 13-17 test."
        )
    )
    parser.add_argument("--config", default="configs/realdisp_rcbd.yaml")
    parser.add_argument("--output", default="outputs/realdisp_fixed_split")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--variant", help="Override model.variant, e.g. rcbd or concat")
    parser.add_argument("--manifest", help="Override data.manifest (useful for smoke tests)")
    parser.add_argument("--epochs", type=int, help="Override training.epochs")
    parser.add_argument("--batch-size", type=int, help="Override training.batch_size")
    parser.add_argument("--num-workers", type=int, help="Override training.num_workers")
    parser.add_argument("--train-participants", nargs="+", default=DEFAULT_TRAIN)
    parser.add_argument("--val-participants", nargs="+", default=DEFAULT_VAL)
    parser.add_argument("--test-participants", nargs="+", default=DEFAULT_TEST)
    parser.add_argument(
        "--refit-full-train",
        action="store_true",
        help="Optional ablation; disabled by default to preserve the published fixed split.",
    )
    args = parser.parse_args()

    config = with_overrides(
        load_config(args.config),
        seed=args.seed,
        **{
            "data.manifest": args.manifest,
            "model.variant": args.variant,
            "training.epochs": args.epochs,
            "training.batch_size": args.batch_size,
            "training.num_workers": args.num_workers,
        },
    )
    frame = load_manifest(config["data"]["manifest"])
    available = set(frame["participant"].astype(str).unique())
    train, val, test = map(
        lambda values: [str(value) for value in values],
        (args.train_participants, args.val_participants, args.test_participants),
    )
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise ValueError("Train, validation, and test participants must be disjoint")
    requested = set(train + val + test)
    if requested != available:
        missing = sorted(requested - available, key=participant_sort_key)
        unexpected = sorted(available - requested, key=participant_sort_key)
        raise ValueError(
            f"Fixed split must cover the manifest exactly; missing={missing}, "
            f"unexpected={unexpected}"
        )

    # run_group_fold derives training participants as all people outside val/test.
    metrics = run_group_fold(
        config,
        test_participants=test,
        val_participants=val,
        output_dir=args.output,
        device_name=args.device,
        refit_full_train=args.refit_full_train,
    )
    output = Path(args.output)
    protocol = {
        "name": "REALDISP fixed user-independent split",
        "config_source": str(Path(args.config).resolve()),
        "manifest": str(Path(config["data"]["manifest"]).resolve()),
        "model_architecture": str(config["model"].get("architecture", "rcbd_har")),
        "model_variant": str(config["model"]["variant"]),
        "num_classes": int(config["data"]["num_classes"]),
        "imu_channels": int(config["data"]["imu_channels"]),
        "train_participants": train,
        "val_participants": val,
        "test_participants": test,
        "refit_full_train": args.refit_full_train,
        "seed": int(config.get("seed", 42)),
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "resolved_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"macro-F1={metrics['macro_f1']:.6%}, accuracy={metrics['accuracy']:.6%}, "
        f"output={output.resolve()}"
    )


if __name__ == "__main__":
    main()
