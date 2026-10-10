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
        description=(
            "Run the label-free placement-distribution calibrated REALDISP "
            "RCBD/prediction-consistency/FAPC comparison."
        )
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
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
    train_participants = [f"subject{value}" for value in range(1, 11)]
    validation_participants = ["subject11", "subject12"]
    profiles = output_root / "placement_profiles_subjects1_10.json"
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
            *train_participants,
        ]
    )
    experiments = [
        ("pd_rcbd", "configs/realdisp_rcbd_rotation_aug.yaml"),
        (
            "pd_prediction_consistency",
            "configs/realdisp_rcbd_rotation_pred_consistency.yaml",
        ),
        ("pd_fapc", "configs/realdisp_fapc_rcbd.yaml"),
    ]
    summary: dict[str, Any] = {
        "method": "Placement-Distribution-Calibrated FAPC (PD-FAPC)",
        "seed": args.seed,
        "protocol": {
            "supervised_train": train_participants,
            "unlabeled_calibration": train_participants,
            "validation": validation_participants,
            "final_test_accessed": False,
            "activity_labels_read_for_calibration": False,
        },
        "placement_profiles": read_json(profiles),
        "experiments": {},
    }
    for name, config in experiments:
        output = output_root / name
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
                str(args.seed),
                "--stress-repeats",
                str(args.stress_repeats),
                "--placement-profile",
                str(profiles),
            ]
        )
        self_output = output / "self_validation_subjects11_12"
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_realdisp_checkpoint.py"),
                "--checkpoint",
                str(output / "best.pt"),
                "--manifest",
                args.self_manifest,
                "--output",
                str(self_output),
                "--device",
                args.device,
                "--participants",
                *validation_participants,
                "--num-workers",
                "4",
            ]
        )
        validation = read_json(output / "validation_metrics.json")
        self_metrics = read_json(self_output / "metrics.json")
        summary["experiments"][name] = {
            "clean_macro_f1": float(validation["clean"]["macro_f1"]),
            "rotation_stress_macro_f1": float(
                validation["rotation_stress"]["macro_f1_mean"]
            ),
            "self_macro_f1": float(self_metrics["macro_f1"]),
            "clean_accuracy": float(validation["clean"]["accuracy"]),
            "self_accuracy": float(self_metrics["accuracy"]),
        }
        with (output_root / "summary.partial.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    prediction = summary["experiments"]["pd_prediction_consistency"]
    fapc = summary["experiments"]["pd_fapc"]
    self_gain = 100.0 * (fapc["self_macro_f1"] - prediction["self_macro_f1"])
    clean_delta = 100.0 * (fapc["clean_macro_f1"] - prediction["clean_macro_f1"])
    summary["frozen_gate"] = {
        "self_gain_pp_vs_pd_prediction_consistency": self_gain,
        "clean_delta_pp_vs_pd_prediction_consistency": clean_delta,
        "minimum_self_gain_pp": 3.0,
        "maximum_clean_drop_pp": 1.0,
        "passes": bool(self_gain >= 3.0 and clean_delta >= -1.0),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
