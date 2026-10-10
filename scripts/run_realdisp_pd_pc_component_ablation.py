from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (256, 3407, 114514)
TRAIN_PARTICIPANTS = tuple(f"subject{value}" for value in range(1, 11))
VALIDATION_PARTICIPANTS = ("subject11", "subject12")
TEST_PARTICIPANTS = tuple(f"subject{value}" for value in range(13, 18))
METHODS = (
    ("clean_rcbd", "configs/realdisp_rcbd.yaml", False),
    ("random_rotation", "configs/realdisp_rcbd_rotation_aug.yaml", False),
    ("pd_rotation_only", "configs/realdisp_pd_rotation_only.yaml", True),
    ("pd_rotation_gain", "configs/realdisp_pd_rotation_gain.yaml", True),
    ("pd_full_no_pc", "configs/realdisp_rcbd_rotation_aug.yaml", True),
    (
        "random_rotation_pc",
        "configs/realdisp_rcbd_rotation_pred_consistency.yaml",
        False,
    ),
    ("pd_pc", "configs/realdisp_rcbd_rotation_pred_consistency.yaml", True),
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


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


def aggregate(output_root: Path) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method, _, _ in METHODS:
        rows = []
        for seed in SEEDS:
            run_root = output_root / method / f"seed{seed}"
            validation_path = run_root / "validation_metrics.json"
            ideal_path = run_root / "final_ideal_subjects13_17" / "metrics.json"
            self_path = run_root / "final_self_subjects13_17" / "metrics.json"
            if not all(path.exists() for path in (validation_path, ideal_path, self_path)):
                continue
            validation = read_json(validation_path)
            ideal = read_json(ideal_path)
            self_metrics = read_json(self_path)
            rows.append(
                {
                    "seed": seed,
                    "validation_ideal_macro_f1": float(validation["clean"]["macro_f1"]),
                    "validation_rotation_stress_macro_f1": float(
                        validation["rotation_stress"]["macro_f1_mean"]
                    ),
                    "final_ideal_macro_f1": float(ideal["macro_f1"]),
                    "final_self_macro_f1": float(self_metrics["macro_f1"]),
                    "final_ideal_accuracy": float(ideal["accuracy"]),
                    "final_self_accuracy": float(self_metrics["accuracy"]),
                }
            )
        method_payload: dict[str, Any] = {"runs": rows}
        if rows:
            for key in (
                "validation_ideal_macro_f1",
                "validation_rotation_stress_macro_f1",
                "final_ideal_macro_f1",
                "final_self_macro_f1",
                "final_ideal_accuracy",
                "final_self_accuracy",
            ):
                method_payload[key] = metric_summary([row[key] for row in rows])
        methods[method] = method_payload
    return {
        "method": "PD-PC predefined component ablation",
        "protocol": {
            "supervised_train": list(TRAIN_PARTICIPANTS),
            "validation": list(VALIDATION_PARTICIPANTS),
            "frozen_final_test": list(TEST_PARTICIPANTS),
            "calibration": list(TRAIN_PARTICIPANTS),
            "calibration_labels_read": False,
            "seeds": list(SEEDS),
            "execution": "strictly sequential and resumable",
        },
        "methods": methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the predefined PD-PC component ablations sequentially."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
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
    profiles = output_root / "placement_profiles_subjects1_10.json"
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

    total = len(METHODS) * len(SEEDS)
    completed = 0
    for method, config, calibrated in METHODS:
        for seed in SEEDS:
            completed += 1
            run_root = output_root / method / f"seed{seed}"
            validation_path = run_root / "validation_metrics.json"
            ideal_output = run_root / "final_ideal_subjects13_17"
            self_output = run_root / "final_self_subjects13_17"
            status = {
                "position": completed,
                "total": total,
                "method": method,
                "seed": seed,
                "state": "running",
            }
            write_json(status, output_root / "queue_status.json")
            print(
                f"[{completed}/{total}] method={method} seed={seed} calibrated={calibrated}",
                flush=True,
            )
            if not validation_path.exists():
                command = [
                    sys.executable,
                    str(ROOT / "scripts" / "run_realdisp_validation_diagnostic.py"),
                    "--config",
                    str(ROOT / config),
                    "--output",
                    str(run_root),
                    "--device",
                    args.device,
                    "--seed",
                    str(seed),
                    "--stress-repeats",
                    str(args.stress_repeats),
                ]
                if calibrated:
                    command.extend(["--placement-profile", str(profiles)])
                run(command)
            checkpoint = run_root / "best.pt"
            for manifest, destination in (
                (args.ideal_manifest, ideal_output),
                (args.self_manifest, self_output),
            ):
                if not (destination / "metrics.json").exists():
                    run(
                        [
                            sys.executable,
                            str(ROOT / "scripts" / "evaluate_realdisp_checkpoint.py"),
                            "--checkpoint",
                            str(checkpoint),
                            "--manifest",
                            manifest,
                            "--output",
                            str(destination),
                            "--device",
                            args.device,
                            "--participants",
                            *TEST_PARTICIPANTS,
                            "--num-workers",
                            "4",
                        ]
                    )
            write_json(aggregate(output_root), output_root / "summary.partial.json")

    summary = aggregate(output_root)
    write_json(summary, output_root / "summary.json")
    write_json(
        {"position": total, "total": total, "state": "completed"},
        output_root / "queue_status.json",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
