from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcbd_har.data.synthetic import (  # noqa: E402
    create_four_limb_synthetic_dataset,
    create_synthetic_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic data for pipeline smoke tests")
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--participants", type=int, default=4)
    parser.add_argument("--windows", type=int, default=24)
    parser.add_argument("--four-limb", action="store_true")
    args = parser.parse_args()
    creator = create_four_limb_synthetic_dataset if args.four_limb else create_synthetic_dataset
    manifest = creator(args.output, args.participants, args.windows)
    print(manifest.resolve())


if __name__ == "__main__":
    main()
