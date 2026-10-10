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
import yaml


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (256, 3407, 114514)
VALID_SIGNAL = tuple(f"subject{value}" for value in range(14, 18))
AS_RELEASED = tuple(f"subject{value}" for value in range(13, 18))
FROZEN_PROFILE_SHA256 = (
    "b05eedb46aa01cff0c2331b09bc3ca4911821a0ef7c8291c0807dd19c555b2b7"
)

METHODS = {
    "uniform_no_pc": {
        "amplitude": "uniform_0_90_degrees",
        "prediction_consistency": False,
        "config": "configs/realdisp_rcbd_rotation_aug.yaml",
        "source": (
            "/root/autodl-tmp/experiments/"
            "realdisp_pd_pc_component_ablation_20260718/random_rotation"
        ),
        "train": False,
    },
    "uniform_pc": {
        "amplitude": "uniform_0_90_degrees",
        "prediction_consistency": True,
        "config": "configs/realdisp_rcbd_rotation_pred_consistency.yaml",
        "source": (
            "/root/autodl-tmp/experiments/"
            "realdisp_pd_pc_component_ablation_20260718/random_rotation_pc"
        ),
        "train": False,
    },
    "empirical_no_pc": {
        "amplitude": "training_subject_ideal_self_empirical_angle",
        "prediction_consistency": False,
        "config": "configs/realdisp_fmc_angle_no_pc.yaml",
        "source": None,
        "train": True,
    },
    "empirical_pc": {
        "amplitude": "training_subject_ideal_self_empirical_angle",
        "prediction_consistency": True,
        "config": "configs/realdisp_fmc_angle_pc.yaml",
        "source": (
            "/root/autodl-tmp/experiments/fmc_pc_completion_20260721/"
            "rcbd_ablation/fmc_angle_pc"
        ),
        "train": False,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def metric_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "std_population": float(array.std(ddof=0)),
    }


def nested_float(payload: dict[str, Any], dotted: str) -> float:
    value: Any = payload
    for key in dotted.split("."):
        value = value[key]
    return float(value)


def checkpoint_identity(checkpoint: Path) -> dict[str, str]:
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
    }


def _same_resolved_path(left: str, right: str) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def cached_metrics_match_checkpoint(
    metrics_path: Path,
    provenance_path: Path,
    checkpoint: Path,
) -> bool:
    """Only reuse metrics bound to the exact current checkpoint file."""
    if not metrics_path.is_file() or not provenance_path.is_file():
        return False
    try:
        metrics = read_json(metrics_path)
        provenance = read_json(provenance_path)
        expected = checkpoint_identity(checkpoint)
        evaluated_checkpoint = metrics["evaluation_protocol"]["checkpoint"]
        provenance_checkpoint = provenance["checkpoint"]
        provenance_sha256 = provenance["checkpoint_sha256"]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        _same_resolved_path(str(evaluated_checkpoint), expected["checkpoint"])
        and _same_resolved_path(str(provenance_checkpoint), expected["checkpoint"])
        and str(provenance_sha256) == expected["checkpoint_sha256"]
    )


def validate_config(method: str, config: dict[str, Any]) -> None:
    expected = METHODS[method]
    training = config["training"]
    loss = config["loss"]
    if float(training["quality_augmentation_probability"]) != 0.5:
        raise ValueError(f"{method}: augmentation probability is not frozen at 0.5")
    actual_pc = float(loss.get("perturbation_prediction_consistency", 0.0))
    expected_pc = 0.1 if expected["prediction_consistency"] else 0.0
    if actual_pc != expected_pc:
        raise ValueError(f"{method}: expected PC={expected_pc}, found {actual_pc}")
    empirical = method.startswith("empirical_")
    if empirical:
        required = {
            "quality_augmentation_profile_sampling": "matched_random",
            "quality_augmentation_profile_rotation": True,
            "quality_augmentation_profile_gain": False,
            "quality_augmentation_profile_noise": False,
        }
        for key, value in required.items():
            if training.get(key) != value:
                raise ValueError(f"{method}: {key} must equal {value!r}")
    elif float(training.get("quality_augmentation_max_degrees", -1.0)) != 90.0:
        raise ValueError(f"{method}: uniform baseline must use 0-90 degrees")


def load_checkpoint_config(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state["config"]


def evaluate(
    checkpoint: Path,
    manifest: Path,
    participants: tuple[str, ...],
    output: Path,
    device: str,
) -> dict[str, Any]:
    metrics_path = output / "metrics.json"
    provenance_path = output / "checkpoint_provenance.json"
    if not cached_metrics_match_checkpoint(
        metrics_path,
        provenance_path,
        checkpoint,
    ):
        run(
            [
                sys.executable,
                str(ROOT / "scripts/evaluate_realdisp_checkpoint.py"),
                "--checkpoint",
                str(checkpoint),
                "--manifest",
                str(manifest),
                "--output",
                str(output),
                "--device",
                device,
                "--participants",
                *participants,
                "--num-workers",
                "4",
            ]
        )
        metrics = read_json(metrics_path)
        evaluated_checkpoint = metrics.get("evaluation_protocol", {}).get(
            "checkpoint"
        )
        expected = checkpoint_identity(checkpoint)
        if evaluated_checkpoint is None or not _same_resolved_path(
            str(evaluated_checkpoint), expected["checkpoint"]
        ):
            raise RuntimeError(
                "Evaluation output is not bound to the requested checkpoint: "
                f"expected {expected['checkpoint']}, found {evaluated_checkpoint!r}"
            )
        write_json(expected, provenance_path)
    return read_json(metrics_path)


def aggregate(runs: dict[str, Any]) -> dict[str, Any]:
    scenarios = ("valid_signal_ideal", "valid_signal_self", "as_released_self")
    metric_paths = {
        "macro_f1": "macro_f1",
        "accuracy": "accuracy",
        "raw_macro_f1": "raw.macro_f1",
        "raw_accuracy": "raw.accuracy",
    }
    methods: dict[str, Any] = {}
    for method, method_runs in runs.items():
        methods[method] = {
            "design": {
                "amplitude": METHODS[method]["amplitude"],
                "prediction_consistency": METHODS[method][
                    "prediction_consistency"
                ],
            },
            "scenarios": {},
            "valid_signal_self_per_participant": {},
            "valid_signal_self_per_class": {},
        }
        for scenario in scenarios:
            methods[method]["scenarios"][scenario] = {
                label: metric_summary(
                    [
                        nested_float(method_runs[str(seed)][scenario], path)
                        for seed in SEEDS
                    ]
                )
                for label, path in metric_paths.items()
            }
        for participant in VALID_SIGNAL:
            methods[method]["valid_signal_self_per_participant"][participant] = {
                label: metric_summary(
                    [
                        float(
                            method_runs[str(seed)]["valid_signal_self"][
                                "per_participant"
                            ][participant][key]
                        )
                        for seed in SEEDS
                    ]
                )
                for label, key in {
                    "macro_f1": "macro_f1",
                    "accuracy": "accuracy",
                }.items()
            }
        for class_id in range(33):
            key = str(class_id)
            methods[method]["valid_signal_self_per_class"][key] = {
                "f1": metric_summary(
                    [
                        float(
                            method_runs[str(seed)]["valid_signal_self"]["per_class"][
                                key
                            ]["f1-score"]
                        )
                        for seed in SEEDS
                    ]
                ),
                "support": int(
                    method_runs[str(SEEDS[0])]["valid_signal_self"]["per_class"][
                        key
                    ]["support"]
                ),
            }
    return {
        "method": "FMC-PC review-requested 2x2 ablation",
        "protocol": {
            "matrix": "uniform_0_90_vs_empirical_angle x PC_off_vs_on",
            "seeds": list(SEEDS),
            "valid_signal": list(VALID_SIGNAL),
            "as_released": list(AS_RELEASED),
            "checkpoint_selection": "subjects11-12 Ideal only",
            "test_based_method_selection": False,
            "same_model_predictions_for_both_test_scopes": True,
        },
        "methods": methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete the frozen FMC-PC 2x2 review ablation.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--placement-profile", required=True, type=Path)
    parser.add_argument("--ideal-manifest", required=True, type=Path)
    parser.add_argument("--self-manifest", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stress-repeats", type=int, default=3)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    for path in (args.placement_profile, args.ideal_manifest, args.self_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(args.placement_profile) != FROZEN_PROFILE_SHA256:
        raise ValueError("Frozen placement profile SHA256 mismatch")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runs: dict[str, Any] = {}
    total = len(METHODS) * len(SEEDS)
    position = 0

    for method, specification in METHODS.items():
        config_path = ROOT / str(specification["config"])
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        validate_config(method, config)
        runs[method] = {}
        for seed in SEEDS:
            position += 1
            write_json(
                {
                    "state": "running",
                    "position": position,
                    "total": total,
                    "method": method,
                    "seed": seed,
                    "updated_utc": utc_now(),
                },
                output_root / "queue_status.json",
            )
            run_root = output_root / method / f"seed{seed}"
            if specification["train"]:
                validation = run_root / "validation_metrics.json"
                if not validation.exists():
                    run(
                        [
                            sys.executable,
                            str(ROOT / "scripts/run_realdisp_validation_diagnostic.py"),
                            "--config",
                            str(config_path),
                            "--output",
                            str(run_root),
                            "--device",
                            args.device,
                            "--seed",
                            str(seed),
                            "--stress-repeats",
                            str(args.stress_repeats),
                            "--placement-profile",
                            str(args.placement_profile),
                        ]
                    )
                checkpoint = run_root / "best.pt"
            else:
                checkpoint = Path(str(specification["source"])) / f"seed{seed}" / "best.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            validate_config(method, load_checkpoint_config(checkpoint))
            scenario_metrics = {
                "valid_signal_ideal": evaluate(
                    checkpoint,
                    args.ideal_manifest,
                    VALID_SIGNAL,
                    run_root / "valid_signal_ideal",
                    args.device,
                ),
                "valid_signal_self": evaluate(
                    checkpoint,
                    args.self_manifest,
                    VALID_SIGNAL,
                    run_root / "valid_signal_self",
                    args.device,
                ),
                "as_released_self": evaluate(
                    checkpoint,
                    args.self_manifest,
                    AS_RELEASED,
                    run_root / "as_released_self",
                    args.device,
                ),
            }
            write_json(
                {
                    "method": method,
                    "seed": seed,
                    "design": {
                        "amplitude": specification["amplitude"],
                        "prediction_consistency": specification[
                            "prediction_consistency"
                        ],
                    },
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                    "config": str(config_path),
                    "config_sha256": sha256(config_path),
                    "placement_profile_sha256": sha256(args.placement_profile),
                    "checkpoint_selection": "subjects11-12 Ideal only",
                    "test_based_method_selection": False,
                },
                run_root / "review_protocol.json",
            )
            runs[method][str(seed)] = scenario_metrics
            write_json(runs, output_root / "runs.partial.json")

    summary = aggregate(runs)
    write_json(summary, output_root / "summary.json")
    write_json(
        {
            "state": "completed",
            "position": total,
            "total": total,
            "updated_utc": utc_now(),
        },
        output_root / "queue_status.json",
    )
    (output_root / "COMPLETED").write_text(utc_now() + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
