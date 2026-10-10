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
SCENARIOS = ("ideal", "self")
AS_RELEASED = tuple(f"subject{value}" for value in range(13, 18))
VALID_SIGNAL = tuple(f"subject{value}" for value in range(14, 18))


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
        description="Re-evaluate frozen REALDISP checkpoints without retraining."
    )
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ideal-manifest", required=True)
    parser.add_argument("--self-manifest", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[256, 3407, 114514])
    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root).resolve()
    output_root = Path(args.output_root).resolve()
    manifests = {"ideal": args.ideal_manifest, "self": args.self_manifest}
    runs: dict[str, Any] = {method: {} for method in METHODS}
    for method in METHODS:
        for seed in args.seeds:
            checkpoint = checkpoint_root / method / f"seed{seed}" / "best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            seed_payload: dict[str, Any] = {}
            for scenario in SCENARIOS:
                released_path = (
                    checkpoint.parent / f"final_{scenario}_subjects13_17" / "metrics.json"
                )
                if not released_path.exists():
                    raise FileNotFoundError(released_path)
                valid_output = output_root / method / f"seed{seed}" / f"valid_{scenario}"
                valid_path = valid_output / "metrics.json"
                if not valid_path.exists():
                    run(
                        [
                            sys.executable,
                            str(ROOT / "scripts/evaluate_realdisp_checkpoint.py"),
                            "--checkpoint",
                            str(checkpoint),
                            "--manifest",
                            manifests[scenario],
                            "--output",
                            str(valid_output),
                            "--device",
                            args.device,
                            "--participants",
                            *VALID_SIGNAL,
                            "--num-workers",
                            "4",
                        ]
                    )
                seed_payload[f"as_released_{scenario}"] = read_json(released_path)
                seed_payload[f"valid_signal_{scenario}"] = read_json(valid_path)
            runs[method][str(seed)] = seed_payload
            write_json(runs, output_root / "runs.partial.json")

    aggregate: dict[str, Any] = {}
    for method, seed_runs in runs.items():
        aggregate[method] = {}
        for cohort in ("as_released", "valid_signal"):
            for scenario in SCENARIOS:
                key = f"{cohort}_{scenario}"
                aggregate[method][key] = {}
                for metric_path in (
                    ("smoothed_macro_f1", "macro_f1"),
                    ("smoothed_accuracy", "accuracy"),
                    ("raw_macro_f1", "raw.macro_f1"),
                    ("raw_accuracy", "raw.accuracy"),
                ):
                    label, dotted = metric_path
                    values = []
                    for seed in args.seeds:
                        value: Any = seed_runs[str(seed)][key]
                        for part in dotted.split("."):
                            value = value[part]
                        values.append(float(value))
                    aggregate[method][key][label] = summarize(values)
    summary = {
        "method": "Frozen-checkpoint smoothed versus raw REALDISP reanalysis",
        "protocol": {
            "retraining": False,
            "as_released_participants": list(AS_RELEASED),
            "post_hoc_valid_signal_participants": list(VALID_SIGNAL),
            "seeds": args.seeds,
            "raw_definition": "window predictions before Gaussian temporal smoothing",
        },
        "aggregate": aggregate,
    }
    write_json(summary, output_root / "summary.json")
    (output_root / "REANALYSIS_COMPLETED").write_text("completed\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
