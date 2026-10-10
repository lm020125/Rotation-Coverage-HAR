from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PARTICIPANTS = tuple(f"subject{value}" for value in range(1, 11))
VALIDATION_PARTICIPANTS = ("subject11", "subject12")
AS_RELEASED_PARTICIPANTS = tuple(f"subject{value}" for value in range(13, 18))
VALID_SIGNAL_PARTICIPANTS = tuple(f"subject{value}" for value in range(14, 18))
METHODS = {
    "matched_random": "configs/realdisp_pd_pc_matched_random.yaml",
    "limb_shuffle": "configs/realdisp_pd_pc_limb_shuffle.yaml",
}
SCENARIOS = {
    "as_released_ideal": ("ideal", AS_RELEASED_PARTICIPANTS),
    "as_released_self": ("self", AS_RELEASED_PARTICIPANTS),
    "valid_signal_ideal": ("ideal", VALID_SIGNAL_PARTICIPANTS),
    "valid_signal_self": ("self", VALID_SIGNAL_PARTICIPANTS),
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def metric_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "values": [float(value) for value in array],
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
    }


def aggregate(output_root: Path, seeds: list[int], profiles: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "method": "PD-PC journal-review strict negative controls",
        "protocol": {
            "supervised_train": list(TRAIN_PARTICIPANTS),
            "validation": list(VALIDATION_PARTICIPANTS),
            "as_released_test": list(AS_RELEASED_PARTICIPANTS),
            "post_hoc_valid_signal_test": list(VALID_SIGNAL_PARTICIPANTS),
            "seeds": seeds,
            "smoothing": "checkpoint-frozen evaluation configuration",
            "raw_metrics": "same predictions before temporal smoothing",
            "execution": "strictly sequential, resumable, stop on first failure",
            "artifact_sha256": {
                "placement_profile": sha256(profiles),
                "placement_source": sha256(ROOT / "src/rcbd_har/placement.py"),
                "engine_source": sha256(ROOT / "src/rcbd_har/engine.py"),
                "runner_source": sha256(Path(__file__).resolve()),
                **{
                    method + "_config": sha256(ROOT / relative_config)
                    for method, relative_config in METHODS.items()
                },
            },
        },
        "methods": {},
    }
    for method in METHODS:
        method_payload: dict[str, Any] = {"runs": {}}
        for seed in seeds:
            run_root = output_root / method / f"seed{seed}"
            seed_payload: dict[str, Any] = {}
            validation_path = run_root / "validation_metrics.json"
            if validation_path.exists():
                validation = read_json(validation_path)
                seed_payload["validation"] = {
                    "smoothed_macro_f1": float(validation["clean"]["macro_f1"]),
                    "raw_macro_f1": float(validation["clean"]["raw"]["macro_f1"]),
                    "rotation_stress_macro_f1": float(
                        validation["rotation_stress"]["macro_f1_mean"]
                    ),
                }
            for scenario in SCENARIOS:
                metrics_path = run_root / scenario / "metrics.json"
                if not metrics_path.exists():
                    continue
                metrics = read_json(metrics_path)
                seed_payload[scenario] = {
                    "smoothed_macro_f1": float(metrics["macro_f1"]),
                    "smoothed_accuracy": float(metrics["accuracy"]),
                    "raw_macro_f1": float(metrics["raw"]["macro_f1"]),
                    "raw_accuracy": float(metrics["raw"]["accuracy"]),
                }
            if seed_payload:
                method_payload["runs"][str(seed)] = seed_payload
        complete_runs = method_payload["runs"]
        method_payload["aggregate"] = {}
        for scenario in SCENARIOS:
            scenario_rows = [
                values[scenario]
                for values in complete_runs.values()
                if scenario in values
            ]
            if len(scenario_rows) != len(seeds):
                continue
            method_payload["aggregate"][scenario] = {
                metric: metric_summary([row[metric] for row in scenario_rows])
                for metric in (
                    "smoothed_macro_f1",
                    "smoothed_accuracy",
                    "raw_macro_f1",
                    "raw_accuracy",
                )
            }
        payload["methods"][method] = method_payload
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run matched-random and limb-shuffled PD-PC controls sequentially."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[256, 3407, 114514])
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
    parser.add_argument("--placement-profile")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    profiles = (
        Path(args.placement_profile).resolve()
        if args.placement_profile
        else output_root / "placement_profiles_subjects1_10.json"
    )
    if not profiles.exists():
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "estimate_realdisp_placement_profiles.py"),
                "--ideal-manifest",
                args.ideal_manifest,
                "--self-manifest",
                args.self_manifest,
                "--output",
                str(profiles),
                "--participants",
                *TRAIN_PARTICIPANTS,
            ]
        )

    total = len(METHODS) * len(args.seeds)
    position = 0
    manifests = {"ideal": args.ideal_manifest, "self": args.self_manifest}
    for method, relative_config in METHODS.items():
        for seed in args.seeds:
            position += 1
            run_root = output_root / method / f"seed{seed}"
            write_json(
                {
                    "position": position,
                    "total": total,
                    "method": method,
                    "seed": seed,
                    "state": "running",
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                },
                output_root / "queue_status.json",
            )
            validation_path = run_root / "validation_metrics.json"
            if not validation_path.exists():
                run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "run_realdisp_validation_diagnostic.py"),
                        "--config",
                        str(ROOT / relative_config),
                        "--output",
                        str(run_root),
                        "--device",
                        args.device,
                        "--seed",
                        str(seed),
                        "--stress-repeats",
                        str(args.stress_repeats),
                        "--placement-profile",
                        str(profiles),
                    ]
                )
            checkpoint = run_root / "best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            for scenario, (manifest_key, participants) in SCENARIOS.items():
                destination = run_root / scenario
                if not (destination / "metrics.json").exists():
                    run(
                        [
                            sys.executable,
                            str(ROOT / "scripts" / "evaluate_realdisp_checkpoint.py"),
                            "--checkpoint",
                            str(checkpoint),
                            "--manifest",
                            manifests[manifest_key],
                            "--output",
                            str(destination),
                            "--device",
                            args.device,
                            "--participants",
                            *participants,
                            "--num-workers",
                            "4",
                        ]
                    )
            write_json(
                aggregate(output_root, args.seeds, profiles),
                output_root / "summary.partial.json",
            )

    summary = aggregate(output_root, args.seeds, profiles)
    write_json(summary, output_root / "summary.json")
    write_json(
        {
            "position": total,
            "total": total,
            "state": "completed",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
        output_root / "queue_status.json",
    )
    (output_root / "QUEUE_COMPLETED").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
