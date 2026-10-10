from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def prerequisite_complete(summary_path: Path, seed: int) -> bool:
    if not summary_path.exists():
        return False
    try:
        summary = read_json(summary_path)
    except (OSError, json.JSONDecodeError):
        return False
    methods = summary.get("methods", {})
    return all(str(seed) in methods.get(name, {}) for name in (
        "prediction_consistency",
        "fapc_rcbd",
    ))


def assert_smoke_diagnostics(history_path: Path) -> dict[str, float]:
    history = read_json(history_path)
    if len(history) != 1:
        raise RuntimeError(f"Expected a one-epoch smoke history, got {len(history)}")
    record = history[0]
    required = (
        "train_gradient_cosine",
        "train_gradient_conflict",
        "train_clean_anchor",
        "train_robust_objective",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise RuntimeError(f"CA-FAPC smoke run lacks diagnostics: {missing}")
    values = {key: float(record[key]) for key in required}
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError(f"Non-finite CA-FAPC smoke diagnostics: {values}")
    if float(record.get("nonfinite_batches", 0.0)) != 0.0:
        raise FloatingPointError(f"CA-FAPC smoke run had non-finite batches: {record}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for frozen seed-114514 baselines, smoke-test CA-FAPC, then run "
            "its untouched REALDISP validation protocol."
        )
    )
    parser.add_argument("--prerequisite-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--prerequisite-seed", type=int, default=114514)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--stress-repeats", type=int, default=3)
    parser.add_argument(
        "--self-manifest",
        default=(
            "/root/autodl-tmp/realdisp_project/data/realdisp_distal_33/"
            "manifest_self.csv"
        ),
    )
    args = parser.parse_args()

    prerequisite = Path(args.prerequisite_summary)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    while not prerequisite_complete(prerequisite, args.prerequisite_seed):
        time.sleep(max(1, args.poll_seconds))

    config = ROOT / "configs" / "realdisp_ca_fapc_rcbd.yaml"
    smoke_output = output_root / f"smoke_seed_{args.seed}"
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_realdisp_validation_diagnostic.py"),
            "--config",
            str(config),
            "--output",
            str(smoke_output),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--epochs",
            "1",
            "--num-workers",
            "4",
            "--stress-repeats",
            "1",
        ]
    )
    smoke_diagnostics = assert_smoke_diagnostics(smoke_output / "history.json")

    experiment_output = output_root / f"seed_{args.seed}" / "ca_fapc"
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_realdisp_validation_diagnostic.py"),
            "--config",
            str(config),
            "--output",
            str(experiment_output),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--stress-repeats",
            str(args.stress_repeats),
        ]
    )
    self_output = experiment_output / "self_validation_subjects11_12"
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_realdisp_checkpoint.py"),
            "--checkpoint",
            str(experiment_output / "best.pt"),
            "--manifest",
            args.self_manifest,
            "--output",
            str(self_output),
            "--device",
            args.device,
            "--participants",
            "subject11",
            "subject12",
            "--num-workers",
            "4",
        ]
    )
    summary = {
        "method": "CA-FAPC",
        "seed": args.seed,
        "prerequisite": {
            "summary": str(prerequisite),
            "seed": args.prerequisite_seed,
            "ordinary_consistency_and_fapc_complete": True,
        },
        "protocol": {
            "train": [f"subject{value}" for value in range(1, 11)],
            "validation": ["subject11", "subject12"],
            "final_test_accessed": False,
        },
        "smoke_diagnostics": smoke_diagnostics,
        "validation": read_json(experiment_output / "validation_metrics.json"),
        "self_validation_subjects11_12": read_json(self_output / "metrics.json"),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"CA-FAPC completed; summary={output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
