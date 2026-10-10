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
SEEDS = (256, 3407, 114514)
VALIDATION_PARTICIPANTS = ("subject11", "subject12")


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


def per_class_f1(payload: dict[str, Any]) -> np.ndarray:
    values = payload["per_class"]
    numeric_labels = sorted(
        (int(label), metrics) for label, metrics in values.items() if str(label).isdigit()
    )
    return np.asarray(
        [float(metrics["f1-score"]) for _, metrics in numeric_labels],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen validation-only admission gate for CR-FMC-PC."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default="configs/realdisp_cr_fmc_pc.yaml")
    parser.add_argument(
        "--placement-profile",
        default=(
            "/root/autodl-tmp/experiments/realdisp_pd_pc_component_ablation_20260718/"
            "placement_profiles_subjects1_10.json"
        ),
    )
    parser.add_argument(
        "--self-manifest",
        default=("/root/autodl-tmp/realdisp_project/data/realdisp_distal_33/manifest_self.csv"),
    )
    parser.add_argument(
        "--baseline-angle-root",
        default=(
            "/root/autodl-tmp/experiments/fmc_pc_completion_20260721/rcbd_ablation/fmc_angle_pc"
        ),
    )
    parser.add_argument(
        "--baseline-self-root",
        default=(
            "/root/autodl-tmp/experiments/cr_fmc_pc_development_20260721/"
            "baseline_angle_pc_validation_self"
        ),
    )
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = (ROOT / args.config).resolve()
    baseline_angle = Path(args.baseline_angle_root).resolve()
    baseline_self = Path(args.baseline_self_root).resolve()

    runs: dict[str, Any] = {}
    for position, seed in enumerate(SEEDS, start=1):
        run_root = output / f"seed{seed}"
        write_json(
            {
                "state": "running",
                "position": position,
                "total": len(SEEDS),
                "seed": seed,
                "final_test_accessed": False,
            },
            output / "queue_status.json",
        )
        validation_metrics = run_root / "validation_metrics.json"
        if not validation_metrics.exists():
            run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_realdisp_validation_diagnostic.py"),
                    "--config",
                    str(config),
                    "--output",
                    str(run_root),
                    "--device",
                    args.device,
                    "--seed",
                    str(seed),
                    "--placement-profile",
                    args.placement_profile,
                ]
            )
        self_metrics = run_root / "validation_self" / "metrics.json"
        if not self_metrics.exists():
            run(
                [
                    sys.executable,
                    str(ROOT / "scripts/evaluate_realdisp_checkpoint.py"),
                    "--checkpoint",
                    str(run_root / "best.pt"),
                    "--manifest",
                    args.self_manifest,
                    "--output",
                    str(run_root / "validation_self"),
                    "--device",
                    args.device,
                    "--participants",
                    *VALIDATION_PARTICIPANTS,
                    "--num-workers",
                    "4",
                ]
            )
        candidate_validation = read_json(validation_metrics)
        candidate_self = read_json(self_metrics)
        base_validation = read_json(baseline_angle / f"seed{seed}" / "validation_metrics.json")
        base_self = read_json(baseline_self / f"seed{seed}" / "metrics.json")
        runs[str(seed)] = {
            "candidate": {
                "ideal_macro_f1": float(candidate_validation["clean"]["macro_f1"]),
                "rotation_stress_macro_f1": float(
                    candidate_validation["rotation_stress"]["macro_f1_mean"]
                ),
                "self_macro_f1": float(candidate_self["macro_f1"]),
                "self_accuracy": float(candidate_self["accuracy"]),
                "per_class_f1": per_class_f1(candidate_self).tolist(),
            },
            "baseline": {
                "ideal_macro_f1": float(base_validation["clean"]["macro_f1"]),
                "rotation_stress_macro_f1": float(
                    base_validation["rotation_stress"]["macro_f1_mean"]
                ),
                "self_macro_f1": float(base_self["macro_f1"]),
                "self_accuracy": float(base_self["accuracy"]),
                "per_class_f1": per_class_f1(base_self).tolist(),
            },
        }
        write_json(runs, output / "runs.partial.json")

    candidate_self = np.asarray([runs[str(seed)]["candidate"]["self_macro_f1"] for seed in SEEDS])
    baseline_self_values = np.asarray(
        [runs[str(seed)]["baseline"]["self_macro_f1"] for seed in SEEDS]
    )
    candidate_ideal = np.asarray([runs[str(seed)]["candidate"]["ideal_macro_f1"] for seed in SEEDS])
    baseline_ideal = np.asarray([runs[str(seed)]["baseline"]["ideal_macro_f1"] for seed in SEEDS])
    candidate_classes = np.asarray(
        [runs[str(seed)]["candidate"]["per_class_f1"] for seed in SEEDS]
    ).mean(axis=0)
    baseline_classes = np.asarray(
        [runs[str(seed)]["baseline"]["per_class_f1"] for seed in SEEDS]
    ).mean(axis=0)
    weak_count = max(1, len(baseline_classes) // 4)
    weak_indices = np.argsort(baseline_classes)[:weak_count]
    checks = {
        "mean_self_gain_at_least_0_5pp": bool(
            candidate_self.mean() >= baseline_self_values.mean() + 0.005
        ),
        "at_least_two_of_three_seeds_improve": bool(
            int((candidate_self > baseline_self_values).sum()) >= 2
        ),
        "ideal_mean_drop_at_most_1pp": bool(candidate_ideal.mean() >= baseline_ideal.mean() - 0.01),
        "baseline_weak_quartile_drop_at_most_1pp": bool(
            candidate_classes[weak_indices].mean() >= baseline_classes[weak_indices].mean() - 0.01
        ),
    }
    gate_pass = all(checks.values())
    summary = {
        "method": "CR-FMC-PC validation-only admission",
        "protocol": {
            "training_participants": [f"subject{value}" for value in range(1, 11)],
            "validation_participants": list(VALIDATION_PARTICIPANTS),
            "excluded_final_test_participants": [f"subject{value}" for value in range(13, 18)],
            "final_test_accessed": False,
            "seeds": list(SEEDS),
            "config": str(config),
            "config_sha256": sha256(config),
            "runner_sha256": sha256(Path(__file__).resolve()),
        },
        "candidate_self_macro_f1": {
            "values": candidate_self.tolist(),
            "mean": float(candidate_self.mean()),
            "std": float(candidate_self.std(ddof=0)),
        },
        "baseline_self_macro_f1": {
            "values": baseline_self_values.tolist(),
            "mean": float(baseline_self_values.mean()),
            "std": float(baseline_self_values.std(ddof=0)),
        },
        "self_gain_pp": float(100.0 * (candidate_self.mean() - baseline_self_values.mean())),
        "candidate_ideal_macro_f1_mean": float(candidate_ideal.mean()),
        "baseline_ideal_macro_f1_mean": float(baseline_ideal.mean()),
        "weak_class_indices": weak_indices.tolist(),
        "candidate_weak_class_f1_mean": float(candidate_classes[weak_indices].mean()),
        "baseline_weak_class_f1_mean": float(baseline_classes[weak_indices].mean()),
        "admission_checks": checks,
        "gate_pass": gate_pass,
        "decision": "retain_and_freeze" if gate_pass else "discard_candidate",
        "runs": runs,
    }
    write_json(summary, output / "summary.json")
    write_json(
        {
            "state": "completed",
            "gate_pass": gate_pass,
            "decision": summary["decision"],
            "final_test_accessed": False,
        },
        output / "queue_status.json",
    )
    marker = "GATE_PASS" if gate_pass else "GATE_FAIL"
    (output / marker).write_text(summary["decision"] + "\n", encoding="utf-8")
    (output / "COMPLETED").write_text("completed\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
