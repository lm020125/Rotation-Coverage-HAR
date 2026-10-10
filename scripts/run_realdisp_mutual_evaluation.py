from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("clean_rcbd", "random_rotation", "pd_pc")
SCENARIOS = ("ideal", "mutual")
TEST_PARTICIPANT = "subject15"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def summarize(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen REALDISP models on mutual displacement for subject15."
    )
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ideal-manifest", required=True)
    parser.add_argument("--mutual-manifest", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[256, 3407, 114514])
    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root).resolve()
    output_root = Path(args.output_root).resolve()
    manifests = {"ideal": args.ideal_manifest, "mutual": args.mutual_manifest}
    runs: dict[str, Any] = {method: {} for method in METHODS}
    for method in METHODS:
        for seed in args.seeds:
            checkpoint = checkpoint_root / method / f"seed{seed}" / "best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            runs[method][str(seed)] = {}
            for scenario in SCENARIOS:
                destination = output_root / method / f"seed{seed}" / scenario
                metrics_path = destination / "metrics.json"
                if not metrics_path.exists():
                    run(
                        [
                            sys.executable,
                            str(ROOT / "scripts/evaluate_realdisp_checkpoint.py"),
                            "--checkpoint",
                            str(checkpoint),
                            "--manifest",
                            manifests[scenario],
                            "--output",
                            str(destination),
                            "--device",
                            args.device,
                            "--participants",
                            TEST_PARTICIPANT,
                            "--num-workers",
                            "4",
                        ]
                    )
                runs[method][str(seed)][scenario] = read_json(metrics_path)
            write_json(runs, output_root / "runs.partial.json")

    aggregate: dict[str, Any] = {}
    for method, seed_runs in runs.items():
        aggregate[method] = {}
        for scenario in SCENARIOS:
            aggregate[method][scenario] = {}
            for label, dotted in (
                ("smoothed_macro_f1", "macro_f1"),
                ("smoothed_accuracy", "accuracy"),
                ("raw_macro_f1", "raw.macro_f1"),
                ("raw_accuracy", "raw.accuracy"),
            ):
                values = []
                for seed in args.seeds:
                    value: Any = seed_runs[str(seed)][scenario]
                    for part in dotted.split("."):
                        value = value[part]
                    values.append(float(value))
                aggregate[method][scenario][label] = summarize(values)
        aggregate[method]["mutual_minus_ideal_pp"] = {
            metric: 100.0
            * (
                aggregate[method]["mutual"][metric]["mean"]
                - aggregate[method]["ideal"][metric]["mean"]
            )
            for metric in ("smoothed_macro_f1", "raw_macro_f1")
        }
    summary = {
        "method": "Frozen-checkpoint REALDISP mutual-displacement evaluation",
        "protocol": {
            "retraining": False,
            "unseen_test_participant": TEST_PARTICIPANT,
            "excluded_from_primary_mutual_estimate": ["subject2", "subject5"],
            "exclusion_reason": "subjects2/5 belong to the supervised training cohort",
            "seeds": args.seeds,
        },
        "aggregate": aggregate,
    }
    write_json(summary, output_root / "summary.json")
    (output_root / "MUTUAL_EVALUATION_COMPLETED").write_text(
        "completed\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
