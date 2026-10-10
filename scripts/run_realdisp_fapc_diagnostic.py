from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen REALDISP validation-only FAPC comparison."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--stress-repeats", type=int, default=3)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    experiments = [
        ("rcbd_rotation", "configs/realdisp_rcbd_rotation_aug.yaml"),
        (
            "prediction_consistency",
            "configs/realdisp_rcbd_rotation_pred_consistency.yaml",
        ),
        ("fapc_rcbd", "configs/realdisp_fapc_rcbd.yaml"),
    ]
    for name, config in experiments:
        output = output_root / name
        command = [
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
        ]
        subprocess.run(command, cwd=ROOT, check=True)

    rows: dict[str, dict[str, float]] = {}
    for name, _ in experiments:
        metrics = load_metrics(output_root / name / "validation_metrics.json")
        if metrics.get("final_test_accessed") is not False:
            raise RuntimeError(f"{name} did not preserve the validation-only protocol")
        rows[name] = {
            "clean_macro_f1": float(metrics["clean"]["macro_f1"]),
            "rotation_stress_macro_f1": float(
                metrics["rotation_stress"]["macro_f1_mean"]
            ),
            "rotation_stress_macro_f1_std": float(
                metrics["rotation_stress"]["macro_f1_std"]
            ),
        }
    baseline = rows["rcbd_rotation"]
    prediction = rows["prediction_consistency"]
    fapc = rows["fapc_rcbd"]
    decision = {
        "clean_drop_pp_vs_rcbd_rotation": 100.0
        * (fapc["clean_macro_f1"] - baseline["clean_macro_f1"]),
        "stress_gain_pp_vs_rcbd_rotation": 100.0
        * (
            fapc["rotation_stress_macro_f1"]
            - baseline["rotation_stress_macro_f1"]
        ),
        "stress_gain_pp_vs_prediction_consistency": 100.0
        * (
            fapc["rotation_stress_macro_f1"]
            - prediction["rotation_stress_macro_f1"]
        ),
    }
    decision["passes_frozen_gate"] = bool(
        decision["clean_drop_pp_vs_rcbd_rotation"] >= -0.5
        and decision["stress_gain_pp_vs_rcbd_rotation"] >= 1.0
        and decision["stress_gain_pp_vs_prediction_consistency"] > 0.0
    )
    summary = {
        "protocol": "REALDISP subjects1-10 train; subjects11-12 validation only",
        "seed": args.seed,
        "final_test_accessed": False,
        "experiments": rows,
        "frozen_gate": {
            "maximum_clean_drop_pp": 0.5,
            "minimum_stress_gain_pp": 1.0,
            "must_beat_prediction_consistency": True,
        },
        "decision": decision,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
