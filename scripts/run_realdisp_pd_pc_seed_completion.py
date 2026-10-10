from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "pd_rcbd": "configs/realdisp_rcbd_rotation_aug.yaml",
    "pd_prediction_consistency": (
        "configs/realdisp_rcbd_rotation_pred_consistency.yaml"
    ),
}
METRICS = ("clean_macro_f1", "rotation_stress_macro_f1", "self_macro_f1")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def collect_result(output: Path) -> dict[str, Any]:
    validation = read_json(output / "validation_metrics.json")
    self_metrics = read_json(output / "self_validation_subjects11_12" / "metrics.json")
    return {
        "clean_macro_f1": float(validation["clean"]["macro_f1"]),
        "rotation_stress_macro_f1": float(
            validation["rotation_stress"]["macro_f1_mean"]
        ),
        "self_macro_f1": float(self_metrics["macro_f1"]),
        "clean_accuracy": float(validation["clean"]["accuracy"]),
        "self_accuracy": float(self_metrics["accuracy"]),
        "self_per_participant_macro_f1": {
            participant: float(metrics["macro_f1"])
            for participant, metrics in self_metrics["per_participant"].items()
        },
    }


def summarize_multiseed(
    methods: dict[str, dict[str, dict[str, Any]]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    seeds = sorted(methods["pd_rcbd"], key=int)
    if set(seeds) != set(methods["pd_prediction_consistency"]):
        raise ValueError("PD-RCBD and PD-PC seeds do not match")
    aggregate: dict[str, Any] = {}
    for method, results in methods.items():
        aggregate[method] = {}
        for metric in METRICS:
            values = np.asarray([results[seed][metric] for seed in seeds], dtype=float)
            aggregate[method][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "values": values.tolist(),
            }
    deltas = {
        seed: {
            "self_gain_pp": 100.0
            * (
                methods["pd_prediction_consistency"][seed]["self_macro_f1"]
                - methods["pd_rcbd"][seed]["self_macro_f1"]
            ),
            "clean_delta_pp": 100.0
            * (
                methods["pd_prediction_consistency"][seed]["clean_macro_f1"]
                - methods["pd_rcbd"][seed]["clean_macro_f1"]
            ),
        }
        for seed in seeds
    }
    self_values = [value["self_gain_pp"] for value in deltas.values()]
    clean_values = [value["clean_delta_pp"] for value in deltas.values()]
    gate = {
        "per_seed_deltas": deltas,
        "mean_self_gain_pp": float(np.mean(self_values)),
        "positive_self_gain_seeds": int(sum(value > 0.0 for value in self_values)),
        "minimum_self_gain_pp": float(min(self_values)),
        "mean_clean_delta_pp": float(np.mean(clean_values)),
        "thresholds": {
            "minimum_mean_self_gain_pp": 3.0,
            "minimum_positive_self_gain_seeds": 2,
            "minimum_single_seed_self_gain_pp": -2.0,
            "minimum_mean_clean_delta_pp": -1.0,
        },
    }
    gate["passes"] = bool(
        gate["mean_self_gain_pp"] >= 3.0
        and gate["positive_self_gain_seeds"] >= 2
        and gate["minimum_self_gain_pp"] >= -2.0
        and gate["mean_clean_delta_pp"] >= -1.0
    )
    return aggregate, gate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Complete frozen PD-RCBD/PD-PC multi-seed REALDISP validation."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--seed3407-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[256, 114514])
    parser.add_argument("--stress-repeats", type=int, default=3)
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
    profile = Path(args.profile).resolve()
    seed3407_root = Path(args.seed3407_root).resolve()
    source_summary = read_json(seed3407_root / "summary.json")
    methods: dict[str, dict[str, dict[str, Any]]] = {
        name: {"3407": dict(source_summary["experiments"][name])}
        for name in METHODS
    }
    for name in METHODS:
        source_output = seed3407_root / name
        source_self = source_output / "self_validation_subjects11_12" / "metrics.json"
        if source_self.exists():
            methods[name]["3407"] = collect_result(source_output)

    summary: dict[str, Any] = {
        "method": "Frozen Placement-Distribution-Calibrated Prediction Consistency",
        "protocol": {
            "train_participants": [f"subject{value}" for value in range(1, 11)],
            "validation_participants": ["subject11", "subject12"],
            "final_test_accessed": False,
            "activity_labels_read_for_calibration": False,
            "seeds": [3407, *args.seeds],
        },
        "profile": {
            "path": str(profile),
            "sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
        },
        "methods": methods,
    }
    for seed in args.seeds:
        for name, config in METHODS.items():
            output = output_root / f"seed_{seed}" / name
            validation_path = output / "validation_metrics.json"
            self_path = output / "self_validation_subjects11_12" / "metrics.json"
            if not (validation_path.exists() and self_path.exists()):
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
                        "--placement-profile",
                        str(profile),
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
                        "subject11",
                        "subject12",
                        "--num-workers",
                        "4",
                    ]
                )
            methods[name][str(seed)] = collect_result(output)
            with (output_root / "summary.partial.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)

    aggregate, gate = summarize_multiseed(methods)
    summary["aggregate"] = aggregate
    summary["frozen_gate"] = gate
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
