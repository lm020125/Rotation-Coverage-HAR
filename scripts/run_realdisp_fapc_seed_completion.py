from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Complete frozen multi-seed FAPC and prediction-consistency experiments."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[256, 114514])
    parser.add_argument("--stress-repeats", type=int, default=3)
    parser.add_argument(
        "--ideal-manifest",
        default=(
            "/root/autodl-tmp/realdisp_project/data/realdisp_distal_33/"
            "manifest_ideal.csv"
        ),
    )
    parser.add_argument(
        "--self-manifest",
        default=(
            "/root/autodl-tmp/realdisp_project/data/realdisp_distal_33/"
            "manifest_self.csv"
        ),
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    methods = [
        (
            "prediction_consistency",
            "configs/realdisp_rcbd_rotation_pred_consistency.yaml",
        ),
        ("fapc_rcbd", "configs/realdisp_fapc_rcbd.yaml"),
    ]
    summary: dict[str, Any] = {
        "protocol": "Frozen configs; no test-guided tuning",
        "seeds": args.seeds,
        "methods": {},
    }
    for seed in args.seeds:
        for name, config in methods:
            output = output_root / f"seed_{seed}" / name
            run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_realdisp_validation_diagnostic.py"),
                    "--config",
                    str(ROOT / config),
                    "--output",
                    str(output),
                    "--device",
                    args.device,
                    "--seed",
                    str(seed),
                    "--stress-repeats",
                    str(args.stress_repeats),
                ]
            )
            evaluations = [
                (
                    "self_validation_subjects11_12",
                    args.self_manifest,
                    ["subject11", "subject12"],
                ),
                (
                    "ideal_final_subjects13_17",
                    args.ideal_manifest,
                    [f"subject{value}" for value in range(13, 18)],
                ),
                (
                    "self_final_subjects13_17",
                    args.self_manifest,
                    [f"subject{value}" for value in range(13, 18)],
                ),
            ]
            result: dict[str, Any] = {
                "validation": read_json(output / "validation_metrics.json")
            }
            for label, manifest, participants in evaluations:
                evaluation_output = output / label
                run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "evaluate_realdisp_checkpoint.py"),
                        "--checkpoint",
                        str(output / "best.pt"),
                        "--manifest",
                        manifest,
                        "--output",
                        str(evaluation_output),
                        "--device",
                        args.device,
                        "--participants",
                        *participants,
                        "--num-workers",
                        "4",
                    ]
                )
                result[label] = read_json(evaluation_output / "metrics.json")
            summary["methods"].setdefault(name, {})[str(seed)] = result
            with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"completed; summary={output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
